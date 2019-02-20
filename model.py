import math
import torch
import torch.nn as nn
from collections import OrderedDict
from modules import MaskConv, BatchRNN, InferenceBatchSoftmax, SequenceWise


def rnn_block(rnn_input_size, rnn_hidden_size, rnn_type, bidirectional, nb_layers):
    """Creates a stack of Batch RNNs with different input_size than hidden_size."""
    rnns = []
    rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                   bidirectional=bidirectional, batch_norm=False)
    rnns.append(('0', rnn))
    for x in range(nb_layers - 1):
        rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional)
        rnns.append(('%d' % (x + 1), rnn))
    return nn.Sequential(OrderedDict(rnns))
        
    
class Head(nn.Module):
    """Shared part of the neural network."""
    def __init__(self, 
                 rnn_type, 
                 rnn_hidden_size,
                 nb_layers, 
                 bidirectional,
                 feature_len,
                 DEBUG):

        super(Head, self).__init__()
        
        self._DEBUG = DEBUG

        # CONV
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        # RNN
        rnn_input_size = feature_len * 8

        self.rnns = rnn_block(rnn_input_size, rnn_hidden_size, rnn_type, bidirectional, nb_layers)


    def forward(self, x, lengths):
        if self._DEBUG:
            print('')
            print('# BEGIN HEAD #')
            print('input', x.size())

        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)

        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = x.transpose(2, 3)
        if self._DEBUG:
            print('after view transpose', x.size())
            
        x, _ = self.conv(x, output_lengths)
        if self._DEBUG:
            print('after conv', x.size())

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        if self._DEBUG:
            print('after view transpose', x.size())

        for rnn in self.rnns:
            x = rnn(x, output_lengths)
        if self._DEBUG:
            print('after rnn', x.size())
    
        self._DEBUG = False
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()


class SpeechToText(nn.Module):
    def __init__(self, 
                 rnn_type, 
                 rnn_hidden_size,
                 nb_layers, 
                 bidirectional,
                 labels,
                 DEBUG):
    
        super(SpeechToText, self).__init__()

        self._DEBUG = DEBUG

        # RNN
        self.rnns = rnn_block(rnn_hidden_size, rnn_hidden_size, rnn_type, bidirectional, nb_layers)
        
        # FULLY CO
        num_classes = len(labels)
        
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()


    def forward(self, x, output_lengths):
        if self._DEBUG:
            print('')
            print('# BEGIN speech to text #')
            print('input', x.size())
            
        for rnn in self.rnns:
            x = rnn(x, output_lengths)
            
        if self._DEBUG:
            print('after rnn', x.size())
        
        x = self.fc(x)
        if self._DEBUG:
            print('after fc', x.size())
        
        x = x.transpose(0, 1)
        if self._DEBUG:
            print('after transpose', x.size())
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        if self._DEBUG:
            print('after softmax', x.size())
            
        x = x.transpose(0, 1)
        if self._DEBUG:
            print('after transpose', x.size())
            
        self._DEBUG = False
        return x


class AccentClassifier(nn.Module):
    def __init__(self,   
                 rnn_type, 
                 rnn_hidden_size,
                 nb_layers, 
                 bidirectional,
                 accents_dict,
                 bottleneck_size,
                 DEBUG):
        
        super(AccentClassifier, self).__init__()
        
        self._DEBUG = DEBUG
                
        # RNN
        self.rnns = rnn_block(rnn_hidden_size, rnn_hidden_size, rnn_type, bidirectional, nb_layers)
            
        # FULLY CO
        num_classes = len(accents_dict)
        
        self.bnf = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, bottleneck_size),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_size),
            nn.Linear(bottleneck_size, num_classes),
            nn.ReLU(),
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, output_lengths):     
        if self._DEBUG:
            print('')
            print('# BEGIN Acc #')
            print('input', x.size())
            
        for rnn in self.rnns:
            x = rnn(x, output_lengths)
            
        if self._DEBUG:
            print('after rnn', x.size())
            
        x = x.mean(dim=0)
        
        if self._DEBUG:
            print('after mean', x.size())
            
        bottleneck = self.bnf(x)
        
        if self._DEBUG:
            print('after bnf', bottleneck.size())
            
        x = self.fc(bottleneck)
        
        if self._DEBUG:
            print('after fc', x.size())
            
        x = self.softmax(x)
        
        if self._DEBUG:
            print('after softmax', x.size())
            
        self._DEBUG = False
        return x, bottleneck


class MultiTask(nn.Module):
    def __init__(self,
                use_mfcc_in=True, 
                use_ivectors_in=True, 
                use_embeddings_in=True,
                use_transcripts_out=True, 
                use_accents_out=True,
                mfcc_size=40,
                ivector_size=100,
                embedding_size=100,
                rnn_type=nn.GRU, 
                labels="abc",
                accents_dict={'uk', 'us'},
                rnn_hidden_size=800, 
                nb_head_layers=2,
                nb_speech_layers=2,
                nb_accents_layers=2,
                bidirectional=True,
                bottleneck_size=256,
                DEBUG=False):
        
        self._meta = {
            'use_mfcc_in': use_mfcc_in, 
            'use_ivectors_in': use_ivectors_in, 
            'use_embeddings_in': use_embeddings_in,
            'use_transcripts_out': use_transcripts_out, 
            'use_accents_out': use_accents_out,
            'mfcc_size': mfcc_size,
            'ivector_size': ivector_size,
            'embedding_size': embedding_size,
            'rnn_type': rnn_type, 
            'labels': labels,
            'accents_dict': accents_dict,
            'rnn_hidden_size': rnn_hidden_size, 
            'nb_head_layers': nb_head_layers,
            'nb_speech_layers': nb_speech_layers,
            'nb_accents_layers': nb_accents_layers,
            'bidirectional': bidirectional,
            'bottleneck_size': bottleneck_size,
            'DEBUG': DEBUG,
        }
        
        super(MultiTask, self).__init__()
            
        feature_len = 0
        feature_len += mfcc_size if use_mfcc_in else 0
        feature_len += ivector_size if use_ivectors_in else 0
        feature_len += embedding_size if use_embeddings_in else 0
            
        self.Head = Head(rnn_type=rnn_type, 
                         rnn_hidden_size=rnn_hidden_size,
                         nb_layers=nb_head_layers, 
                         bidirectional=bidirectional,
                         feature_len=feature_len,
                         DEBUG=DEBUG)
            
        if self._meta['use_transcripts_out']:
            self.SpeechToText = SpeechToText(rnn_type=rnn_type, 
                                             rnn_hidden_size=rnn_hidden_size,
                                             nb_layers=nb_speech_layers, 
                                             bidirectional=bidirectional,
                                             labels=labels,
                                             DEBUG=DEBUG)
            
        if self._meta['use_accents_out']:
            self.AccentClassifier = AccentClassifier(rnn_type=rnn_type, 
                                        rnn_hidden_size=rnn_hidden_size,
                                        nb_layers=nb_accents_layers, 
                                        bidirectional=bidirectional,
                                        accents_dict=accents_dict,
                                        bottleneck_size=bottleneck_size,
                                        DEBUG=DEBUG)
        
    def forward(self, x, lengths):
        x, out_len = self.Head(x, lengths)
        x_stt, x_acc = None, None
        
        if self._meta['use_transcripts_out']:
            x_stt = self.SpeechToText(x, out_len)
            
        if self._meta['use_accents_out']:
            x_acc, bnf = self.AccentClassifier(x, out_len)
            
        return x_stt, x_acc, out_len, bnf
    
    
    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
    
    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        meta = package['meta']
        model = cls(
            use_mfcc_in = meta['use_mfcc_in'], 
            use_ivectors_in = meta['use_ivectors_in'], 
            use_embeddings_in = meta['use_embeddings_in'],
            use_transcripts_out = meta['use_transcripts_out'], 
            use_accents_out = meta['use_accents_out'],
            mfcc_size = meta['mfcc_size'],
            ivector_size = meta['ivector_size'],
            embedding_size = meta['embedding_size'],
            rnn_type = meta['rnn_type'], 
            labels = meta['labels'],
            accents_dict = meta['accents_dict'],
            rnn_hidden_size = meta['rnn_hidden_size'], 
            nb_head_layers = meta['nb_head_layers'],
            nb_speech_layers = meta['nb_speech_layers'],
            nb_accents_layers = meta['nb_accents_layers'],
            bidirectional = meta['bidirectional'],
            bottleneck_size = meta['bottleneck_size'],
            DEBUG = meta['DEBUG'],
        )
        model.load_state_dict(package['state_dict'])
        return model
        
    @staticmethod
    def serialize(model, 
                  path='./__temp__',
                  save=True,
                  exp_name=None,
                  optimizer=None, 
                  epoch=None,
                  train_losses=None,
                  test_losses=None,
                  text_train_losses=None,
                  text_test_losses=None,
                  text_wers=None,
                  accent_train_losses=None,
                  accent_test_losses=None,
                  accent_accuracies=None):
        
        """Saves the model in a packaged form. Also returns the package.
        Use the load_model class method to recreate a model from a package."""
        
        package = {
            'state_dict': model.state_dict(),
            'meta': model._meta
        }
        
        if exp_name is not None:
            package['exp_name'] = exp_name
        if optimizer is not None:
            package['optimizer'] = optimizer
        if epoch is not None:
            package['epoch'] = epoch
        if train_losses is not None:
            package['train_losses'] = train_losses
        if test_losses is not None:
            package['test_losses'] = test_losses
        if text_train_losses is not None:
            package['text_train_losses'] = text_train_losses
        if text_test_losses is not None:
            package['text_test_losses'] = text_test_losses
        if text_wers is not None:
            package['text_wers'] = text_wers
        if accent_train_losses is not None:
            package['accent_train_losses'] = accent_train_losses
        if accent_test_losses is not None:
            package['accent_test_losses'] = accent_test_losses
        if accent_accuracies is not None:
            package['accent_accuracies'] = accent_accuracies
            
        if save:
            torch.save(package, str(path) + '.pth')
            
        return package