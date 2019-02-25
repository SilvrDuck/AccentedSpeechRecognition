from dataloader import MultiDataset, MultiDataLoader
from focalloss import FocalLoss
from warpctc_pytorch import CTCLoss
from model import MultiTask
from decoder import GreedyDecoder, BeamCTCDecoder
from training import train, test
import torch
import torch.nn as nn
from os import makedirs
from tensorboardX import SummaryWriter
from pathlib import Path
import math
from utils import now_str
import gc

    
def run_experiment(_exp_name,
                   _epochs,
                   _train_manifest, 
                   _test_manifest, 
                   _labels, 
                   _use_mfcc_in, 
                   _use_ivectors_in, 
                   _use_embeddings_in, 
                   _use_transcripts_out, 
                   _use_accents_out, 
                   _batch_size, 
                   _num_workers,
                   _mfcc_size,
                   _ivector_size,
                   _embedding_size,
                   _rnn_type, 
                   _rnn_hidden_size, 
                   _nb_head_layers,
                   _nb_speech_layers,
                   _nb_accents_layers,
                   _bidirectional,
                   _losses_mix,
                   _learning_rate,
                   _lm_path,
                   _decoder_alpha, 
                   _decoder_beta,
                   _decoder_cutoff_top_n,
                   _decoder_beam_width,
                   _cuda,
                   _tensorboard_path,
                   _saved_models_path,
                   _bottleneck_size,
                   _accent_loss):
    
    print(f'\n##### Running experiment {_exp_name} #####')
    
    # Tools to log values
    results_dict = {}
    results_dict['train_loss'] = []
    results_dict['train_loss_text'] = []
    results_dict['train_loss_accent'] = []   
    results_dict['test_loss'] = []
    results_dict['test_loss_text'] = []
    results_dict['test_loss_accent'] = []
    results_dict['test_wer'] = []
    results_dict['test_accent_acc'] = []
    
    tb_path = Path(_tensorboard_path) / _exp_name
    makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(tb_path)

    ### DATA LOADING

    # Training set
    train_dataset = MultiDataset(_train_manifest,
                                 _labels, 
                                 use_mfcc_in=_use_mfcc_in, 
                                 use_ivectors_in=_use_ivectors_in, 
                                 use_embeddings_in=_use_embeddings_in, 
                                 embedding_size=_embedding_size,
                                 use_transcripts_out=_use_transcripts_out, 
                                 use_accents_out=_use_accents_out)
    
    train_loader = MultiDataLoader(train_dataset, 
                                       batch_size=_batch_size, 
                                       shuffle=True, 
                                       num_workers=_num_workers)
        
    # Testing set
    test_dataset = MultiDataset(_test_manifest,
                                _labels, 
                                use_mfcc_in=_use_mfcc_in, 
                                use_ivectors_in=_use_ivectors_in, 
                                use_embeddings_in=_use_embeddings_in, 
                                embedding_size=_embedding_size,
                                use_transcripts_out=_use_transcripts_out, 
                                use_accents_out=_use_accents_out)

    test_loader = MultiDataLoader(test_dataset, 
                                      batch_size=_batch_size, 
                                      shuffle=True, 
                                      num_workers=_num_workers)

    
    ### CREATE MODEL
    
    model = MultiTask(use_mfcc_in = _use_mfcc_in, 
                      use_ivectors_in = _use_ivectors_in, 
                      use_embeddings_in = _use_embeddings_in,
                      use_transcripts_out = _use_transcripts_out, 
                      use_accents_out = _use_accents_out,
                      mfcc_size = _mfcc_size,
                      ivector_size = _ivector_size,
                      embedding_size = _embedding_size,
                      rnn_type = _rnn_type, 
                      labels = _labels,
                      accents_dict = train_dataset.accent_dict,
                      rnn_hidden_size = _rnn_hidden_size, 
                      nb_head_layers = _nb_head_layers,
                      nb_speech_layers = _nb_speech_layers,
                      nb_accents_layers = _nb_accents_layers,
                      bidirectional = _bidirectional,
                      bottleneck_size = _bottleneck_size,
                      DEBUG=False)
    if _cuda:
        model = model.cuda()
        
    print(model, '\n')
    print('Model parameters counts:', MultiTask.get_param_size(model), '\n')
    
    ### OPTIMIZER, CRITERION, DECODER
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=_learning_rate)
    
    # Criterion
    if _use_accents_out:
        if _accent_loss == 'focal':
            AccLoss = FocalLoss()
        elif _accent_loss == 'CE':
            AccLoss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Loss {_accent_loss} for accent_loss is unknown. Please use either "focal" or "CE".')
    
    if not _use_transcripts_out: # only accent classification
        criterion = AccLoss
    elif not _use_accents_out: # only text recognition
        criterion = CTCLoss()
    else: # both tasks
        criterion = (CTCLoss(), FocalLoss())
        
    # Decoder
    if _use_transcripts_out:
        decoder = BeamCTCDecoder(_labels, 
                                 lm_path=_lm_path,
                                 alpha=_decoder_alpha, 
                                 beta=_decoder_beta,
                                 cutoff_top_n=_decoder_cutoff_top_n,
                                 cutoff_prob=_decoder_cutoff_top_n,
                                 beam_width=_decoder_beam_width, 
                                 num_processes=_num_workers)
        
        target_decoder = GreedyDecoder(_labels)
    else:
        decoder, target_decoder = None, None
    
    
    ### EPOCHS
    best_wer = math.inf
    best_acc = 0
    
    for epoch in range(1, _epochs + 1):
        ### TRAIN    
        print(f'Epoch {epoch} training')
        train_results = train(model, train_loader, criterion, optimizer, losses_mix=_losses_mix)
        train_loss, train_loss_text, train_loss_accent = train_results        

        results_dict['train_loss'].append(train_loss)
        results_dict['train_loss_text'].append(train_loss_text)
        results_dict['train_loss_accent'].append(train_loss_accent)
        print(f'Epoch {epoch} training loss: {train_loss}')
        
        ### TEST
        print(f'Epoch {epoch} testing')
        test_results = test(model, test_loader, criterion, decoder, target_decoder, losses_mix=_losses_mix)
        test_loss, test_loss_text, test_loss_accent, test_wer, test_accent_acc = test_results
        
        results_dict['test_loss'].append(test_loss)
        results_dict['test_loss_text'].append(test_loss_text)
        results_dict['test_loss_accent'].append(test_loss_accent)
        results_dict['test_wer'].append(test_wer)
        results_dict['test_accent_acc'].append(test_accent_acc)
        print(f'Epoch {epoch} testing loss: {test_loss}')
        
        # Add values to tensorboard
        for key, results in results_dict.items():
            tb_writer.add_scalar(key, results[-1], epoch)
        
        #Save model if it is best
        save_new=False
        if _use_transcripts_out:
            if test_wer < best_wer:
                save_new = True                
                best_wer = test_wer
        else:
            if test_accent_acc > best_acc:
                save_new = True
                best_acc = test_accent_acc
                
        if save_new:
            MultiTask.serialize(model, 
                                Path(_saved_models_path) / _exp_name,
                                save=True,
                                exp_name=_exp_name,
                                optimizer=optimizer, 
                                epoch=epoch,
                                train_losses=results_dict['train_loss'],
                                test_losses=results_dict['test_loss'],
                                text_train_losses=results_dict['train_loss_text'],
                                text_test_losses=results_dict['test_loss_text'],
                                text_wers=results_dict['test_wer'],
                                accent_train_losses=results_dict['train_loss_accent'],
                                accent_test_losses=results_dict['test_loss_accent'],
                                accent_accuracies=results_dict['test_accent_acc'])
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    ## end of run_experiment ##

    
### MAIN

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DeepSpeech model information')
    parser.add_argument('--train', action='store_true', help='Uses the train set instead of the dev set.')
    parser.add_argument('--epochs', default=None, type=int, help='Number of training epochs')
    parser.add_argument('--patch_path', default='experiments.cfg', type=str, help='Path to experiment list')
    args = parser.parse_args()
    
    DEV = not args.train
    PATCH_PATH = args.patch_path
    EPOCHS = args.epochs
    
    import config
    confs = config.Config()
    
    for conf in confs.patch_config(PATCH_PATH):
        exp_name = conf['exp_name_prefix']
        exp_name += '_DEV' if DEV else '_TRAIN'
        exp_name += '__in'
        exp_name += '_mfcc' if conf['use_mfcc_in'] else '' 
        exp_name += '_ivect' if conf['use_ivectors_in'] else '' 
        exp_name += '_emb' if conf['use_embeddings_in'] else ''
        exp_name += '__out'
        exp_name += '_transcripts' if conf['use_transcripts_out'] else '' 
        exp_name += f'_accents-mix{conf["losses_mix"]}-{conf["accent_loss"]}' if conf['use_accents_out'] else ''
        exp_name += f'__nblyrs-head-{conf["nb_head_layers"]}'
        exp_name += f'-speech-{conf["nb_speech_layers"]}'
        exp_name += f'-accent-{conf["nb_accents_layers"]}'
        exp_name += f'__bnf-{conf["bottleneck_size"]}'
        exp_name += f'__{now_str()}'
        
        train_manifest = conf['dev_manifest'] if DEV else conf['train_manifest']
        epochs = EPOCHS if EPOCHS is not None else conf['epochs']
        
        try:
            run_experiment(_exp_name = exp_name,
                           _epochs = epochs,
                           _train_manifest = train_manifest, 
                           _test_manifest = conf['test_manifest'], 
                           _labels = conf['labels'], 
                           _use_mfcc_in = conf['use_mfcc_in'], 
                           _use_ivectors_in = conf['use_ivectors_in'], 
                           _use_embeddings_in = conf['use_embeddings_in'], 
                           _use_transcripts_out = conf['use_transcripts_out'], 
                           _use_accents_out = conf['use_accents_out'], 
                           _batch_size = conf['batch_size'], 
                           _num_workers = conf['num_workers'],
                           _mfcc_size = conf['mfcc_size'],
                           _ivector_size = conf['ivector_size'],
                           _embedding_size = conf['embedding_size'],
                           _rnn_type = conf['rnn_type'], 
                           _rnn_hidden_size = conf['rnn_hidden_size'], 
                           _nb_head_layers = conf['nb_head_layers'],
                           _nb_speech_layers = conf['nb_speech_layers'],
                           _nb_accents_layers = conf['nb_accents_layers'],
                           _bidirectional = conf['bidirectional'],
                           _losses_mix = conf['losses_mix'],
                           _learning_rate = conf['learning_rate'],
                           _lm_path = conf['lm_path'],
                           _decoder_alpha = conf['decoder_alpha'], 
                           _decoder_beta = conf['decoder_beta'],
                           _decoder_cutoff_top_n = conf['decoder_cutoff_top_n'],
                           _decoder_beam_width = conf['decoder_beam_width'],
                           _cuda = conf['cuda'],
                           _tensorboard_path = conf['tensorboard_path'],
                           _saved_models_path = conf['saved_models_path'],
                           _bottleneck_size = conf['bottleneck_size'],
                           _accent_loss = conf['accent_loss'])
        
        except Exception as e:
            print(f'Error occured in run {exp_name}:', e)