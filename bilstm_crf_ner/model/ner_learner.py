""" Works with pytorch 0.4.0 """

import spacy
from .core import *
from .data_utils import pad_sequences, minibatches, get_chunks
from .crf import CRF
from .general_utils import Progbar
from torch.optim.lr_scheduler import StepLR

if os.name == "posix": from allennlp.modules.elmo import Elmo, batch_to_ids # AllenNLP is currently only supported on linux


class NERLearner(object):
    """
    NERLearner class that encapsulates a pytorch nn.Module model and ModelData class
    Contains methods for training a testing the model
    """
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.logger = self.config.logger
        self.model = model
        self.model_path = config.dir_model
        self.use_elmo = config.use_elmo


        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

        self.criterion = CRF(self.config.ntags)
        self.optimizer = optim.Adam(self.model.parameters())

        if self.use_elmo:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        else:
            self.load_emb()

        if USE_GPU:
            self.use_cuda = True
            self.logger.info("GPU found.")
            self.model = model.cuda()
            self.criterion = self.criterion.cuda()
            if self.use_elmo:
                self.elmo = self.elmo.cuda()
                print("Moved elmo to cuda")
        else:
            self.model = model.cpu()
            self.use_cuda = False
            self.logger.info("No GPU found.")

    def get_model_path(self, name):
        return os.path.join(self.model_path,name)+'.h5'

    def get_layer_groups(self, do_fc=False):
        return children(self.model)

    def freeze_to(self, n):
        c=self.get_layer_groups()
        for l in c:
            set_trainable(l, False)
        for l in c[n:]:
            set_trainable(l, True)

    def unfreeze(self):
        self.freeze_to(0)

    def save(self, name=None):
        if not name:
            name = self.config.ner_model_path
        save_model(self.model, self.get_model_path(name))
        self.logger.info(f"Saved model at {self.get_model_path(name)}")

    def load_emb(self):
        self.model.emb.weight = nn.Parameter(T(self.config.embeddings))
        self.model.emb.weight.requires_grad = False
        self.logger.info('Loading pretrained word embeddings')

    def load(self, fn=None):
        if not fn: fn = self.config.ner_model_path
        fn = self.get_model_path(fn)
        load_ner_model(self.model, fn, strict=True)
        self.logger.info(f"Loaded model from {fn}")

    def batch_iter(self, train, batch_size, return_lengths=False, shuffle=False, sorter=False):
        """
        Builds a generator from the given dataloader to be fed into the model

        Args:
            train: DataLoader
            batch_size: size of each batch
            return_lengths: if True, generator returns a list of sequence lengths for each
                            sample in the batch
                            ie. sequence_lengths = [8,7,4,3]
            shuffle: if True, shuffles the data for each epoch
            sorter: if True, uses a sorter to shuffle the data

        Returns:
            nbatches: (int) number of batches
            data_generator: batch generator yielding
                                dict inputs:{'word_ids' : np.array([[padded word_ids in sent1], ...])
                                             'char_ids': np.array([[[padded char_ids in word1_sent1], ...],
                                                                    [padded char_ids in word1_sent2], ...],
                                                                    ...])}
                                labels: np.array([[padded label_ids in sent1], ...])
                                sequence_lengths: list([len(sent1), len(sent2), ...])

        """
        nbatches = (len(train) + batch_size - 1) // batch_size

        def  data_generator():
            while True:
                if shuffle: train.shuffle()
                elif sorter==True and train.sorter: train.sort()

                for i, (words, labels) in enumerate(minibatches(train, batch_size)):

                    # perform padding of the given data
                    if self.config.use_chars:
                        char_ids, word_ids = zip(*words)
                        word_ids, sequence_lengths = pad_sequences(word_ids, 1)
                        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                        nlevels=2)

                    else:
                        word_ids, sequence_lengths = pad_sequences(words, 0)

                    if self.use_elmo:
                        word_ids = words

                    if labels:
                        labels, _ = pad_sequences(labels, 0)
                        # if categorical
                        ## labels = [to_categorical(label, num_classes=len(train.tag_itos)) for label in labels]

                    # build dictionary
                    inputs = {
                        "word_ids": np.asarray(word_ids)
                    }

                    if self.config.use_chars:
                        inputs["char_ids"] = np.asarray(char_ids)

                    if return_lengths:
                        yield(inputs, np.asarray(labels), sequence_lengths)

                    else:
                        yield (inputs, np.asarray(labels))

        return (nbatches, data_generator())


    def fine_tune(self, train, dev=None):
        """
        Fine tune the NER model by freezing the pre-trained encoder and training the newly
        instantiated layers for 1 epochs
        """
        self.logger.info("Fine Tuning Model")
        self.fit(train, dev, epochs=1, fine_tune=True)


    def fit(self, train, dev=None, epochs=None, fine_tune=False):
        """
        Fits the model to the training dataset and evaluates on the validation set.
        Saves the model to disk
        """
        if not epochs:
            epochs = self.config.nepochs
        batch_size = self.config.batch_size

        nbatches_train, train_generator = self.batch_iter(train, batch_size,
                                                          return_lengths=True)
        if dev:
            nbatches_dev, dev_generator = self.batch_iter(dev, batch_size,
                                                      return_lengths=True)

        scheduler = StepLR(self.optimizer, step_size=1, gamma=self.config.lr_decay)

        if not fine_tune: self.logger.info("Training Model")

        f1s = []

        for epoch in range(epochs):
            scheduler.step()
            self.train(epoch, nbatches_train, train_generator, fine_tune=fine_tune)

            if dev:
                f1 = self.test(nbatches_dev, dev_generator, fine_tune=fine_tune)

            # Early stopping
            if len(f1s) > 0:
                if f1 < max(f1s[max(-self.config.nepoch_no_imprv, -len(f1s)):]): #if sum([f1 > f1s[max(-i, -len(f1s))] for i in range(1,self.config.nepoch_no_imprv+1)]) == 0:
                    print("No improvement in the last 3 epochs. Stopping training")
                    break
            else:
                f1s.append(f1)

        if fine_tune:
            self.save(self.config.ner_ft_path)
        else :
            self.save(self.config.ner_model_path)


    def train(self, epoch, nbatches_train, train_generator, fine_tune=False):
        self.logger.info('\nEpoch: %d' % epoch)
        self.model.train()
        if not self.use_elmo: self.model.emb.weight.requires_grad = False

        train_loss = 0
        correct = 0
        total = 0
        total_step = None

        prog = Progbar(target=nbatches_train)

        for batch_idx, (inputs, targets, sequence_lengths) in enumerate(train_generator):

            if batch_idx == nbatches_train: break
            if inputs['word_ids'].shape[0] == 1:
                self.logger.info('Skipping batch of size=1')
                continue

            total_step = batch_idx
            targets = T(targets, cuda=self.use_cuda).transpose(0,1).contiguous()
            self.optimizer.zero_grad()

            if self.use_elmo:
                sentences = inputs['word_ids']
                character_ids = batch_to_ids(sentences)
                if self.use_cuda:
                    character_ids = character_ids.cuda()
                embeddings = self.elmo(character_ids)
                word_input = embeddings['elmo_representations'][0]
                word_input, targets = Variable(word_input, requires_grad=False), \
                                      Variable(targets)
                inputs = (word_input)

            else:
                word_input = T(inputs['word_ids'], cuda=self.use_cuda)
                char_input = T(inputs['char_ids'], cuda=self.use_cuda)
                word_input, char_input, targets = Variable(word_input, requires_grad=False), \
                                                  Variable(char_input, requires_grad=False),\
                                                  Variable(targets)
                inputs = (word_input, char_input)


            outputs = self.model(inputs)

            # Create mask
            if self.use_elmo:
                mask = Variable(embeddings['mask'].transpose(0,1))
                if self.use_cuda:
                    mask = mask.cuda()
            else:
                mask = create_mask(sequence_lengths, targets, cuda=self.use_cuda)

            # Get CRF Loss
            loss = -1*self.criterion(outputs, targets, mask=mask)
            loss.backward()
            self.optimizer.step()

            # Callbacks
            train_loss += loss.item()
            predictions = self.criterion.decode(outputs, mask=mask)
            masked_targets = mask_targets(targets, sequence_lengths)

            t_ = mask.type(torch.LongTensor).sum().item()
            total += t_
            c_ = sum([1 if p[i] == mt[i] else 0 for p, mt in zip(predictions, masked_targets) for i in range(len(p))])
            correct += c_

            prog.update(batch_idx + 1, values=[("train loss", loss.item())], exact=[("Accuracy", 100*c_/t_)])

        self.logger.info("Train Loss: %.3f, Train Accuracy: %.3f%% (%d/%d)" %(train_loss/(total_step+1), 100.*correct/total, correct, total) )


    def test(self, nbatches_val, val_generator, fine_tune=False, evaluate=False):
        self.model.eval()
        accs = []
        test_loss = 0
        correct_preds = 0
        total_correct = 0
        total_preds = 0
        total_step = None

        for batch_idx, (inputs, targets, sequence_lengths) in enumerate(val_generator):
            if batch_idx == nbatches_val: break
            if inputs['word_ids'].shape[0] == 1:
                self.logger.info('Skipping batch of size=1')
                continue

            total_step = batch_idx
            targets = T(targets, cuda=self.use_cuda).transpose(0,1).contiguous()
            input_tokens = inputs["word_ids"]

            if self.use_elmo:
                sentences = inputs['word_ids']
                character_ids = batch_to_ids(sentences)
                if self.use_cuda:
                    character_ids = character_ids.cuda()
                embeddings = self.elmo(character_ids)
                word_input = embeddings['elmo_representations'][1]
                word_input, targets = Variable(word_input, requires_grad=False), \
                                      Variable(targets)
                inputs = (word_input)

            else:
                word_input = T(inputs['word_ids'], cuda=self.use_cuda)
                char_input = T(inputs['char_ids'], cuda=self.use_cuda)
                word_input, char_input, targets = Variable(word_input, requires_grad=False), \
                                                  Variable(char_input, requires_grad=False),\
                                                  Variable(targets)
                inputs = (word_input, char_input)

            outputs = self.model(inputs)

            # Create mask
            if self.use_elmo:
                mask = Variable(embeddings['mask'].transpose(0,1))
                if self.use_cuda:
                    mask = mask.cuda()
            else:
                mask = create_mask(sequence_lengths, targets, cuda=self.use_cuda)

            # Get CRF Loss
            loss = -1*self.criterion(outputs, targets, mask=mask)

            # Callbacks
            test_loss += loss.item()
            predictions = self.criterion.decode(outputs, mask=mask)
            if evaluate:
                write_test_preds(
                    input_tokens,
                    predictions,
                    self.config.vocab_tags,
                    self.config.filename_test_preds
                )
            masked_targets = mask_targets(targets, sequence_lengths)

            for lab, lab_pred in zip(masked_targets, predictions):

                accs    += [1 if a==b else 0 for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        self.logger.info("Val Loss : %.3f, Val Accuracy: %.3f%%, Val F1: %.3f%%" %(test_loss/(total_step+1), 100*acc, 100*f1))
        return 100*f1

    def evaluate(self,test):
        batch_size = self.config.batch_size
        nbatches_test, test_generator = self.batch_iter(test, batch_size,
                                                        return_lengths=True)
        self.logger.info('Evaluating on test set')
        self.test(nbatches_test, test_generator, fine_tune=False, evaluate=True)

    def predict_batch(self, words, sequence_lengths):
        self.model.eval()
        if len(words) == 1:
            mult = np.ones(2).reshape(2, 1).astype(int)

        if self.use_elmo:
            sentences = words
            character_ids = batch_to_ids(sentences)
            if self.use_cuda:
                character_ids = character_ids.cuda()
            embeddings = self.elmo(character_ids)
            word_input = embeddings['elmo_representations'][1]
            word_input = Variable(word_input, requires_grad=False)

            if len(words) == 1:
                word_input = ((torch.tensor(mult)*word_input.transpose(0,1)).transpose(0,1).contiguous()).type(torch.FloatTensor)

            word_input = T(word_input, cuda=self.use_cuda)
            inputs = (word_input)

        else:
            #char_ids, word_ids = zip(*words)
            char_ids = [[c[0] for c in s] for s in words]
            word_ids = [[x[1] for x in s] for s in words]
            word_ids, sequence_lengths = pad_sequences(word_ids, 1)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                                   nlevels=2)
            word_ids = np.asarray(word_ids)
            char_ids = np.asarray(char_ids)

            if len(words) == 1:
                word_ids = mult*word_ids
                char_ids = (mult*char_ids.transpose(1,0,2)).transpose(1,0,2)
            word_input = T(word_ids, cuda=self.use_cuda)
            char_input = T(char_ids, cuda=self.use_cuda)

            word_input, char_input = Variable(word_input, requires_grad=False), \
                                     Variable(char_input, requires_grad=False)

            inputs = (word_input, char_input)


        outputs = self.model(inputs)

        predictions = self.criterion.decode(outputs)
        predictions = [p[:i] for p, i in zip(predictions, sequence_lengths)]
        return predictions

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string)

        Returns:
            preds: list of tags (string)

        """

        sequence_lengths = [len(p) for p in words_raw]

        if self.use_elmo:
            words = words_raw
        else:
            words = [[self.config.processing_word(w) for w in s] for s in words_raw]

        pred_ids = self.predict_batch(words, sequence_lengths)
        preds = [[self.idx_to_tag[idx.item() if isinstance(idx, torch.Tensor) else idx]  for idx in s] for s in pred_ids]
        return preds


def create_mask(sequence_lengths, targets, cuda, batch_first=False):
    """ Creates binary mask """
    mask = Variable(torch.ones(targets.size()).type(torch.ByteTensor))
    if cuda: mask = mask.cuda()

    for i,l in enumerate(sequence_lengths):
        if batch_first:
            if l < targets.size(1):
                mask.data[i, l:] = 0
        else:
            if l < targets.size(0):
                mask.data[l:, i] = 0

    return mask


def mask_targets(targets, sequence_lengths, batch_first=False):
    """ Masks the targets """
    if not batch_first:
         targets = targets.transpose(0,1)
    t = []
    for l, p in zip(targets,sequence_lengths):
        t.append(l[:p].data.tolist())
    return t

def write_test_preds(input_tokens, predictions, tags, filename):
    """Join Tokens and it's predictions and save it to a file"""
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    with open(filename, "a") as f:
        assert len(input_tokens) == len(predictions)
        for inp_tokens, inp_predictions in zip(input_tokens, predictions):
            assert len(inp_tokens) == len(inp_predictions)
            for token, label in zip(inp_tokens, inp_predictions):
                f.write("{} {}\n".format(token, idx_to_tag[int(label)]))
            f.write("\n")


