import pymsteams
import numpy as np
import torch
import time
import json
import spacy
import os
import sys
import getopt

from Model import Autoencoder, TOKEN_LENGTH, EMBEDDING_SIZE


def count_logfile_line(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b:
                break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count


def prepare(tokenizer, samples):
    embeddings = []
    sentences = []
    for i, sample in enumerate(samples):
        sentence = tokenizer(sample)
        sentences.append(sentence)
        embeddings.append([])
        for y, token in enumerate(sentence):
            if y < TOKEN_LENGTH:
                embeddings[i].append(np.array(token.vector[:EMBEDDING_SIZE]))
        for _w in range(TOKEN_LENGTH-len(embeddings[i])):
            embeddings[i].append(np.zeros(EMBEDDING_SIZE))

    embeddings = np.array(embeddings)

    return torch.from_numpy(
                embeddings.reshape(
                    [len(samples), 1, TOKEN_LENGTH, EMBEDDING_SIZE]
                )).double(), samples, sentences


def getNewLines():
    cur = json.loads(open('model/cursor.dat', 'r').read())
    lines = []

    for key, value in cur.items():
        l = count_logfile_line(key)
        if l < value - 10:
            value = 0
        if l > value:
            print('new lines in ', key, ' : ', value, ' -> ', l)
            lines = lines + os.popen('tail -n '
                                     + str(l-value) + ' '+key).readlines()
            cur[key] = l

    for i in range(len(lines)):
        if lines[i].find('[') > -1 and lines[i].find(']') > -1:
            lines[i] = lines[i][lines[i].find(']')+1:]

    with open('model/cursor.dat', 'w') as f:
        f.write(json.dumps(cur))

    return lines


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def prepareComponents(teamsHookUrl):
    print('-Create Model')
    model = Autoencoder()
    model = model.double()
    print('-Load')

    device = get_device()
    try:
        if device == 'cpu':
            model.load_state_dict(
                torch.load('model/AE_prod.pt', map_location=torch.device('cpu'))
            )
        else:
            model.load_state_dict(torch.load('model/AE_prod.pt'))
    except Exception:
        print("Can't load checkpoint. Abort.")
        exit(1)
    tokenizer = spacy.load('en_core_web_md')

    if teamsHookUrl is not None:
        teamsMessaging = pymsteams.connectorcard(teamsHookUrl)
    else:
        teamsMessaging = None

    model.to(device)
    print('-Ready!')
    return model, device, tokenizer, teamsMessaging


def monitoringLoop(model, device, tokenizer, teamsMessaging, threshold):
    print('-Currently Monitoring ...')
    while True:
        samples, original, tokens = prepare(tokenizer, getNewLines())
        if len(samples) > 0:

            samples = samples.to(device)
            outputs = model(samples)

            samples = samples.detach().numpy()
            outputs = outputs.detach().numpy()

            for i in range(len(samples)):

                mse = np.mean(np.power(samples[i][0] - outputs[i][0], 2))
                print(mse)
                if mse > threshold:
                    print(mse, ' - ', original[i])
                    with open('data/alerts.txt', 'a') as f:
                        f.write(original[i])
                        if teamsMessaging is not None:
                            teamsMessaging.text(original[i])
                            teamsMessaging.send()
                else:
                    with open('data/standard.txt', 'a') as f:
                        f.write(original[i])

        time.sleep(5)


if __name__ == "__main__":
    teamsHookUrl = None
    threshold = None

    opts, args = getopt.getopt(sys.argv[1:], "h:t:",
                               ["hookteams=", "threshold="])

    for opt, arg in opts:
        if opt in ("-h", "--hookteams"):
            teamsHookUrl = arg
        if opt in ("-t", "--threshold"):
            threshold = float(arg)

    if threshold == "":
        print('Pass threshold with -t or --threshold')
        sys.exit(2)

    if teamsHookUrl == "":
        print('No Teams hook url, messaging will be disabled')

    model, device, tokenizer, teamsMessaging = prepareComponents(teamsHookUrl)
    monitoringLoop(model, device, tokenizer, teamsMessaging, threshold)
