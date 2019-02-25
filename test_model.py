from model import MultiTask
from training import test
from dataloader import MultiDataset, MultiDataLoader
from focalloss import FocalLoss
from warpctc_pytorch import CTCLoss
from decoder import GreedyDecoder, BeamCTCDecoder


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
        
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DeepSpeech model information')
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('--lm', type=str, help='Path to language model')
    args = parser.parse_args()
    
    import config
    confs = config.Config()
    
    model, __ = MultiTask.load_model(args.model_path)
    if confs['cuda']:
        model = model.cuda()
    
    # Criterion
    if model._meta['use_accents_out']:
        if confs['accent_loss'] == 'focal':
            AccLoss = FocalLoss()
        elif confs['accent_loss'] == 'CE':
            AccLoss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Loss {confs["accent_loss"]} for accent_loss is unknown. Please use either "focal" or "CE".')
    
    if not model._meta['use_transcripts_out']: # only accent classification
        criterion = AccLoss
    elif not model._meta['use_accents_out']: # only text recognition
        criterion = CTCLoss()
    else: # both tasks
        criterion = (CTCLoss(), FocalLoss())
        
        
    # Decoders
    
    lm_path = args.lm if args.lm else confs['lm_path']
    if model._meta['use_transcripts_out']:
        decoder = BeamCTCDecoder(confs['labels'], 
                                 lm_path=lm_path,
                                 alpha=confs['decoder_alpha'], 
                                 beta=confs['decoder_beta'],
                                 cutoff_top_n=confs['decoder_cutoff_top_n'],
                                 cutoff_prob=confs['decoder_cutoff_top_n'],
                                 beam_width=confs['decoder_beam_width'], 
                                 num_processes=confs['num_workers'])
        
        target_decoder = GreedyDecoder(confs['labels'])
    else:
        decoder, target_decoder = None, None
    
    # Results
    results = {}
    for manifest in confs['testing_manifests']:
        print(f'Testing {manifest}')
        results[manifest.split('/')[-1]] = result_for_manifest(model, criterion, manifest, decoder, target_decoder, confs['batch_size'], confs['num_workers'])
        
    for name, res in results.items():
        print(f'\nResults for {name}:')
        print('; '.join([f'{k}: {v:.3f}' for k, v in res.items()]))