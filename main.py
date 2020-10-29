import argparse

from Config.config import Config
from DataProcessing.data_read import *
from Model.Domain_model import DomainModel
from Vocab.vocab import *
from Train.train import train
from Model.BertModel import MyBertModel
from transformers import BertTokenizer

if __name__ == '__main__':

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    tokenizer = BertTokenizer.from_pretrained('E:\my_nlp\SMP\SMP-4.1\RoBERTa_zh_Large_PyTorch/.')

    # seed
    random_seed(520)

    # gpu
    gpu = torch.cuda.is_available()
    if gpu:
        print('The train will be using GPU.')
    else:
        print('The train will be using CPU.')
    print('CuDNN', torch.backends.cudnn.enabled)

    # config
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file', type=str, default='./Config/config.ini')
    args = arg_parser.parse_args()
    config = Config(args.config_file)
    if gpu:
        config.add_args('Train', 'use_cuda', 'True')
        config.add_args('Train', 'fp16', 'False')

    word_data_loader = ReadData(config)

    usual_tra_data_set, virus_tra_data_set, usual_dev_data_set, virus_dev_data_set, usual_vat_data, virus_vat_data \
        = word_data_loader.read_data(tokenizer)
    usual_eval_data, virus_eval_data = word_data_loader.read_eval_data(tokenizer)

    # vocab
    if os.path.isfile(config.fact_word_src_vocab):
        tag_vocab = read_pkl(config.fact_word_tag_vocab)
    else:
        if not os.path.isdir(config.save_vocab_path):
            os.makedirs(config.save_vocab_path)
        tag_vocab = FactWordTagVocab([usual_tra_data_set, virus_tra_data_set])
        pickle.dump(tag_vocab, open(config.fact_word_tag_vocab, 'wb'))

    domain_vocab = DomainVocab()

    bert_model = MyBertModel(config)
    domain_model = DomainModel(config)

    if config.use_cuda:
        bert_model = bert_model.cuda()
        domain_model = domain_model.cuda()

    # train
    train(bert_model, domain_model, usual_tra_data_set, virus_tra_data_set, usual_dev_data_set, virus_dev_data_set,
          usual_vat_data, virus_vat_data, tag_vocab, config, domain_vocab, tokenizer, usual_eval_data, virus_eval_data)
