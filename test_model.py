from __future__ import print_function
from model import MultiTask
from training import test
from dataloader import MultiDataset, MultiDataLoader
import torch.nn as nn
import torch
from focalloss import FocalLoss
from warpctc_pytorch import CTCLoss
from decoder import GreedyDecoder, BeamCTCDecoder
import sys
import sys
from pathlib import Path

PRINT_LATEX_TABLE = True

manual_seed = 666
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all
print(f'Using torch manual seed {manual_seed}.')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

    
def result_for_manifest(model, criterion, manifest, decoder, target_decoder, batch_size, num_workers):
    ### LOADER
    test_dataset = MultiDataset(manifest,
                                model._meta['labels'], 
                                use_mfcc_in=model._meta['use_mfcc_in'], 
                                use_ivectors_in=model._meta['use_ivectors_in'], 
                                use_embeddings_in=model._meta['use_embeddings_in'], 
                                embedding_size=model._meta['embedding_size'],
                                use_transcripts_out=model._meta['use_transcripts_out'], 
                                use_accents_out=model._meta['use_accents_out'])

    test_loader = MultiDataLoader(test_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=True, 
                                      num_workers=num_workers)
    
    ### TEST
    test_results = test(model, test_loader, criterion, decoder, target_decoder)
    test_loss, test_loss_text, test_loss_accent, test_wer, test_accent_acc = test_results

    results_dict = {}
    
    if test_wer != -1:
        results_dict['WER'] = test_wer
    if test_accent_acc != -1:
        results_dict['Accent accuracy'] = test_accent_acc
    
    return results_dict
        

def main(model_path, confs): 
    model, __ = MultiTask.load_model(model_path)
    if confs['cuda']:
        model = model.cuda()
    
    
    if not model._meta['use_transcripts_out']: # only accent classification
        criterion = nn.CrossEntropyLoss()
    elif not model._meta['use_accents_out']: # only text recognition
        criterion = CTCLoss()
    else: # both tasks
        criterion = (CTCLoss(), nn.CrossEntropyLoss())
        
    
    # Results
    results = {}
    for manifest, lm in confs['testing_manifests']:
        eprint(f'\n### Testing {manifest.split("/")[-1]} for model {Path(model_path).stem.split("_")[0]}')
        
        # Decoder
        if model._meta['use_transcripts_out']:
            decoder = BeamCTCDecoder(confs['labels'], 
                                     lm_path=lm,
                                     alpha=confs['decoder_alpha'], 
                                     beta=confs['decoder_beta'],
                                     cutoff_top_n=confs['decoder_cutoff_top_n'],
                                     cutoff_prob=confs['decoder_cutoff_top_n'],
                                     beam_width=confs['decoder_beam_width'], 
                                     num_processes=confs['num_workers'])

            target_decoder = GreedyDecoder(confs['labels'])
        else:
            decoder, target_decoder = None, None
        
        # Test
        results[manifest.split('/')[-1]] = result_for_manifest(model, criterion, manifest, decoder, target_decoder, confs['batch_size'], confs['num_workers'])
        
        
    if not PRINT_LATEX_TABLE:
        print(f'Model: {model_path.split("/")[-1]}')
        for name, res in results.items():
            print(f'\nResults for {name}:')
            print('; '.join([f'{k}: {v:.3f}' for k, v in res.items()]))
    else:
        print(' & '.join(['model']+list([k[:-4] for k in results.keys()])))
        val_dict = {}
        for k in list(results.values())[0].keys():
            val_dict[k] = []
        for res in results.values():
            [val_dict[k].append(f'{v:.1f}') for k, v in res.items()]
        for val in val_dict.values():
            print(' & '.join([Path(model_path).stem.split('_')[0]]+val)+r' \\')
        
if __name__ == '__main__':
    import config
    confs = config.Config()
    
    args = sys.argv[1:]
    
    if PRINT_LATEX_TABLE:
        eprint('\nLatex output selected, change PRINT_LATEX_TABLE in script to False for regular output.')
      
    for model_path in args:
        main(model_path, confs)