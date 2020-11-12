from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
# from model.ent_model import EntModel
# from model.ent_learner import EntLearner


def main():
    # create instance of config
    config = Config()
    if config.use_elmo: config.processing_word = None

    #build model
    model = NERModel(config)

    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter, config.use_crf)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter, config.use_crf)

    learn = NERLearner(config, model)
    learn.fit(train, dev)


if __name__ == "__main__":
    main()

