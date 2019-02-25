import json

import torch
import torch.nn as nn
import numpy as np
from utils import tile
from torch.utils.data import DataLoader, Dataset


### DATASET

class MultiDataset(Dataset):
    """Defines an iterator over the dataset. This class is intended to be used with
    the MultiDataLoader class."""

    def __init__(self, manifest, labels, manifest_separator=',',
               use_mfcc_in=True, use_ivectors_in=False, use_embeddings_in=False,
               embedding_size=100, use_transcripts_out=True, use_accents_out=False):
        """
        Allows to chose what will be trained on, and what are the outputs.
        At least on input and one output is needed.
        Default configuration is regular MFCCs to text.
        
        Manifest should be csv type file with following row for each sample:
        mfcc_path, ivector_path, embedding_path, transcripts_path, accent_label
        (Column can remain empty if not used, but must be present.)
        
        Scripts to create the database and manifest from audio and text in the scripts folder.
        """
        
        assert(any([use_mfcc_in, use_ivectors_in, use_embeddings_in])), 'MultiDataset config needs at least one input set to True'
        assert(any([use_transcripts_out, use_accents_out])), 'MultiDataset config needs at least one output set to True'
        assert(not use_transcripts_out or use_mfcc_in), 'Can’t do speech to text without mfcc.'
        
        super(MultiDataset, self).__init__()
        
        self.config = {}
        self.config['use_mfcc_in']=use_mfcc_in
        self.config['use_ivectors_in']=use_ivectors_in
        self.config['use_embeddings_in']=use_embeddings_in
        self.config['embedding_size']=embedding_size
        self.config['use_transcripts_out']=use_transcripts_out
        self.config['use_accents_out']=use_accents_out

        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        
        with open(manifest) as f:
            self.samples = [x.strip().split(manifest_separator) for x in f.readlines()]
            
        self.accent_dict = self.make_accent_dict(self.samples)
            
    def __getitem__(self, index):
        """Unused features are set to None for the Dataloader. Returns torch tensors."""
        mfcc_path, ivector_path, embedding_path, transcript_path, accent_label = self.samples[index]
        mfcc, ivector, embedding, parsed_transcript, accent = None, None, None, None, None
        
        def load_array(path):
            with open(path) as f:
                array = json.load(f)
            return torch.FloatTensor(array)
            
        # Inputs
        if self.config['use_mfcc_in']:
            mfcc = load_array(mfcc_path)
            
        if self.config['use_ivectors_in']:
            ivector = load_array(ivector_path)
            
        if self.config['use_embeddings_in']:
            new_embedding_path = []
            for split in embedding_path.split('/'):
                new = split if 'embedding' not in split else ''.join([split, '_', str(self.config['embedding_size'])])
                new_embedding_path.append(new)
            new_embedding_path = '/'.join(new_embedding_path)
            embedding = torch.load(new_embedding_path, map_location=lambda storage, loc: storage)
            # map_location and loc are there to load the embedding on the CPU
            
        # Outputs
        if self.config['use_transcripts_out']:
            parsed_transcript = self.parse_transcript(transcript_path)
            
        if self.config['use_accents_out']:
            accent = self.accent_dict[accent_label]
            accent = torch.LongTensor([accent])
        
        return mfcc, ivector, embedding, parsed_transcript, accent
        
        
    def parse_transcript(self, transcript_path):
        """Maps a text to integers using the given labels_map."""
        
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
            
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript
    
    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def make_accent_dict(samples):
        acc_set = set()
        for __, __, __, __, accent in samples:
            acc_set.add(accent)
        enum = enumerate(sorted(acc_set)) # sorted set for consistant results
        return {acc: i for i, acc in enum}
    
    
### DATALOADER

# Collate function for the MultiDataLoader
def collate_fn(batch):
    """This function takes list of samples and assembles a batch. 
    It is intended to used in PyTorch DataLoader."""
    
    mfccs, ivectors, embeddings, transcripts, accents = list(zip(*batch))
    
    def exists(list_): 
        """Checks if we are not getting a list of None"""
        return list_[0] is not None
    
    ## Lens
    if exists(mfccs): 
        inputs_lens = torch.IntTensor([len(m) for m in mfccs])
    elif exists(ivectors):
        inputs_lens = torch.IntTensor([len(i) for i in ivectors])
    else:
        inputs_lens = torch.IntTensor([1] * len(batch))
        
    # Sorting order (needs to be descending in lens for the padder)
    inputs_lens, sorted_idx = inputs_lens.sort(descending=True)
        
    if exists(transcripts):
        transcripts_lens = torch.IntTensor([len(t) for t in transcripts])
        transcripts_lens = transcripts_lens[sorted_idx]
    else:
        transcripts_lens = None
        
    ## Inputs
    inputs = []
    if exists(mfccs):
        inputs.append(nn.utils.rnn.pad_sequence(mfccs, batch_first=True))
        
    if exists(ivectors):
        ivect = nn.utils.rnn.pad_sequence(ivectors, batch_first=True)
        if exists(mfccs): # The ivector resolution is 10 times lower than the mfccs', so we expand them.
            ivect = tile(ivect, 1, 10) 
            ivect = ivect[:, :inputs[0].size(1), :]
        inputs.append(ivect)
        
    if exists(embeddings):
        emb = torch.cat(embeddings)
        emb = emb.view(emb.size(0), 1, emb.size(1))
        if exists(mfccs) or exists(ivectors): 
            # tile embeddings to fit either mfccs or ivectors size if they are present
            emb = tile(emb, 1, inputs[0].size(1))   
        inputs.append(emb)
        
    inputs = torch.cat(inputs, dim=2)
    inputs = inputs[sorted_idx]
    
    ## Outputs
    if exists(transcripts):
        if inputs.size(0) == 1: # bugfix for when only one sample
            transcripts = [transcripts]
        transcripts = np.asarray(transcripts)[sorted_idx] # dtype=object because some transcripts were loaded with wrong type (Int64). TODO fix.
        transcripts = torch.IntTensor([t for trs in transcripts for t in trs]) 
        # we need text targets as one concatenated vector
        
    if exists(accents):
        accents = torch.cat(accents)[sorted_idx]
    else:
        accents = None

    return inputs, inputs_lens, transcripts, transcripts_lens, accents

class MultiDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for SpeechDatasets.
        """
        super(MultiDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_fn