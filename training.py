from tqdm import tqdm
import torch
import numpy as np

def get_mixed_loss(criterion, out_text, out_accent, out_lens, accents, transcripts, transcripts_lens, mix=0.5, corrective_coef=1000):
    loss, loss_text, loss_accent = None, None, None
 
    if out_text is None:
        loss_accent = criterion(out_accent, accents)
        loss = loss_accent
    elif out_accent is None:
        loss_text = criterion(out_text, transcripts, out_lens, transcripts_lens)
        loss = loss_text
    else:
        loss_text = criterion[0](out_text, transcripts, out_lens, transcripts_lens)
        loss_accent = criterion[1](out_accent, accents)
        if loss_accent.is_cuda:
            loss_text = loss_text.cuda()
        loss = mix * loss_text + (1 - mix) * loss_accent * corrective_coef
        
    return loss, loss_text, loss_accent


### TRAINING

def train(model, train_loader, criterion, optimizer, losses_mix=None):

    epoch_losses = []
    epoch_losses_text = []
    epoch_losses_accent = []

    model.train()

    for data in tqdm(train_loader, total=len(train_loader)):
        inputs, inputs_lens, transcripts, transcripts_lens, accents = data

        if next(model.parameters()).is_cuda:
            inputs = inputs.cuda()
            inputs_lens = inputs_lens.cuda()

            if accents is not None:
                accents = accents.cuda()

        out_text, out_accent, out_lens, __ = model(inputs, inputs_lens)

        loss, loss_text, loss_accent = get_mixed_loss(criterion, out_text, out_accent, 
                                                      out_lens, accents, transcripts, 
                                                      transcripts_lens, losses_mix)

        epoch_losses.append(loss)
        epoch_losses_text.append(loss_text)
        epoch_losses_accent.append(loss_accent)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = lambda l: sum(l) / len(train_loader) if l[0] is not None else -1

    epoch_loss = average_loss(epoch_losses)
    epoch_loss_text = average_loss(epoch_losses_text)
    epoch_loss_accent = average_loss(epoch_losses_accent)

    return epoch_loss, epoch_loss_text, epoch_loss_accent


### TESTING

def check_wer(transcripts, transcripts_lens, out, out_lens, decoder, target_decoder):
    split_transcripts = []
    offset = 0
    for size in transcripts_lens:
        split_transcripts.append(transcripts[offset:offset + size])
        offset += size
        
    decoded_output, _ = decoder.decode(out.data.transpose(0,1), out_lens)
    target_strings = target_decoder.convert_to_strings(split_transcripts)
           
    #if True:
    #    print('targets', targets)
    #    print('split_targets', split_targets)
    #    print('out', out)
    #    print('output_len', output_len)
    #    print('decoded', decoded_output)
    #    print('target', target_strings)
        
    wer, cer = 0, 0
    for x in range(len(target_strings)):
        transcript, reference = decoded_output[x][0], target_strings[x][0]
        wer += decoder.wer(transcript, reference) / float(len(reference.split()))
        #cer += decoder.cer(transcript, reference) / float(len(reference))
    wer /= len(target_strings)
    return wer * 100


def check_acc(accents, out):
    out_arg = np.argmax(out, axis=1)
    diff = torch.eq(out_arg, accents.cpu())
    acc = torch.sum(diff)
    return acc / len(accents) * 100


def test(model, test_loader, criterion, decoder, target_decoder, losses_mix):
    with torch.no_grad():
        model.eval()

        epoch_losses = []
        epoch_losses_text = []
        epoch_losses_accent = []

        epoch_wers = []
        epoch_accent_accs = []

        for data in tqdm(test_loader, total=len(test_loader)):
            inputs, inputs_lens, transcripts, transcripts_lens, accents = data

            if next(model.parameters()).is_cuda:
                inputs = inputs.cuda()
                inputs_lens = inputs_lens.cuda()

                if accents is not None:
                    accents = accents.cuda()

            out_text, out_accent, out_lens, __ = model(inputs, inputs_lens)

            loss, loss_text, loss_accent = get_mixed_loss(criterion, out_text, out_accent, 
                                                          out_lens, accents, transcripts, 
                                                          transcripts_lens, losses_mix)

            if out_text is not None:
                wer = check_wer(transcripts, transcripts_lens, 
                                out_text, out_lens, decoder, target_decoder)
            else:
                wer = None

            if out_accent is not None:
                accent_acc = check_acc(accents, out_accent)
            else:
                accent_acc = None

            epoch_losses.append(loss)
            epoch_losses_text.append(loss_text)
            epoch_losses_accent.append(loss_accent)

            epoch_wers.append(wer)
            epoch_accent_accs.append(accent_acc)

        average_loss = lambda l: sum(l) / len(test_loader) if l[0] is not None else -1

        epoch_loss = average_loss(epoch_losses)
        epoch_loss_text = average_loss(epoch_losses_text)
        epoch_loss_accent = average_loss(epoch_losses_accent)

        epoch_wer = average_loss(epoch_wers)
        epoch_accent_acc = average_loss(epoch_accent_accs)

    return epoch_loss, epoch_loss_text, epoch_loss_accent, epoch_wer, epoch_accent_acc
