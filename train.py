import argparse
import json
import os
import time
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from datasets import load_dataset
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from model import SentenceVAE
from text_dataset import DataCollator, TextDataset
from utils import to_var, idx2word, expierment_name


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    RANDOM_SEED = 42

    dataset = load_dataset("yelp_polarity", split="train")
    TRAIN_SIZE = len(dataset) - 2_000
    VALID_SIZE = 1_000
    TEST_SIZE = 1_000

    train_test_split = dataset.train_test_split(train_size=TRAIN_SIZE, seed=RANDOM_SEED)
    train_dataset = train_test_split["train"]
    test_val_dataset = train_test_split["test"].train_test_split(train_size=VALID_SIZE, test_size=TEST_SIZE,
                                                                 seed=RANDOM_SEED)
    val_dataset, test_dataset = test_val_dataset["train"], test_val_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    datasets = OrderedDict()
    datasets['train'] = TextDataset(train_dataset, tokenizer, args.max_sequence_length, not args.disable_sent_tokenize)
    datasets['valid'] = TextDataset(val_dataset, tokenizer, args.max_sequence_length, not args.disable_sent_tokenize)
    if args.test:
        datasets['text'] = TextDataset(test_dataset, tokenizer, args.max_sequence_length, not args.disable_sent_tokenize)

    print(f"Loading {args.model_name} model. Setting {args.trainable_layers} trainable layers.")
    encoder = AutoModel.from_pretrained(args.model_name, return_dict=True)
    if not args.train_embeddings:
        for p in encoder.embeddings.parameters():
            p.requires_grad = False
    encoder_layers = encoder.encoder.layer
    if args.trainable_layers > len(encoder_layers):
        warnings.warn(f"You are asking to train {args.trainable_layers} layers, but this model has only {len(encoder_layers)}")
    for layer in range(len(encoder_layers) - args.trainable_layers):
        for p in encoder_layers[layer].parameters():
            p.requires_grad = False
    params = dict(
        vocab_size=datasets['train'].vocab_size,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        max_sequence_length=args.max_sequence_length
    )
    model = SentenceVAE(encoder=encoder, tokenizer=tokenizer, **params)

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    with open(os.path.join(save_model_path, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    def kl_anneal_function(anneal_function, step, k, x0):
        if step <= x0:
            return args.initial_kl_weight
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0-2500))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')
    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    params = [
        {'params': model.encoder.parameters(), 'lr': args.encoder_learning_rate},
        {
            'params': [
                *model.decoder_rnn.parameters(),
                *model.hidden2mean.parameters(),
                *model.hidden2logv.parameters(),
                *model.latent2hidden.parameters(),
                *model.outputs2vocab.parameters()
            ]
        }
    ]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=(split == 'train'),
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available(),
                collate_fn=DataCollator(tokenizer)
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['input'], batch['attention_mask'], batch['length'])

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step, args.k, args.x0)

                loss = (NLL_loss + KL_weight * KL_loss) / batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # bookkeepeing
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO" % split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss" % split.upper(), NLL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss" % split.upper(), KL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight" % split.upper(), KL_weight,
                                      epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                          % (split.upper(), iteration, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size,
                          KL_loss.item()/batch_size, KL_weight))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].tolist(), tokenizer=tokenizer)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, args.epochs, tracker['ELBO'].mean()))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences, the encoded latent space and generated sequences
            if split == 'valid':
                samples, _ = model.inference(z=tracker['z'])
                generated_sents = idx2word(samples.tolist(), tokenizer)
                sents = [{'original': target, 'generated': generated}
                         for target, generated in zip(tracker['target_sents'], generated_sents)]
                dump = {'sentences': sents, 'z': tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json' % epoch), 'w') as dump_file:
                    json.dump(dump, dump_file, indent=3)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_sequence_length', type=int, default=256)
    parser.add_argument('--disable_sent_tokenize', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-elr', '--encoder_learning_rate', type=float, default=2e-5)
    parser.add_argument('-l2', '--weight_decay', type=float, default=0.001)

    parser.add_argument('-mn', '--model_name', type=str, default='roberta-base',
                        help='Name of transformers model to use')
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=64)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-tl', '--trainable_layers', type=int, default=2,
                        help='Number of layers to optimize in transformer encoder')
    parser.add_argument('-te', '--train_embeddings', action='store_true',
                        help='Whether or not to train embedding layer')

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)
    parser.add_argument('-w0', '--initial_kl_weight', type=float, default=0.001)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
