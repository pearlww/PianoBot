
import numpy as np
import config

#Returns an array of integer sequences [ [part1], [part2], ...]

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100

START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
}

vocab_size = config.vocab_size

#max_time in milliseconds
def split_encoding(int_array, max_time):
    max_time = max_time / 10 #In time shift units
    time = 0
    note_array=[False]*RANGE_NOTE_ON  #Note on/Note off array
    res=[]
    part=[]
    times=[]
    
    #print("Int array: ")
    #print(str(int_array))
    
    for event in int_array:
        
        if event < START_IDX['note_off']:
            #note_on event
            note_array[event] = True
            part.append(event)
        elif event < START_IDX['time_shift']:
            #note off event
            if note_array[event - START_IDX['note_off']]:
                note_array[event - START_IDX['note_off']] = False
                part.append(event)
            #Note that, in the else clause, we keep the note off 
            #AND we don't add that unnecessary event to the part
            #Can happen after splits
        elif event < START_IDX['velocity']:
            #time shift event
            shift = event - START_IDX['time_shift']
            time += shift #Increment time
            if time >= max_time:
                #This last shift overlaps the splitting point
                finish_time = shift - (time - max_time)
                #Append the correct duration to the part
                part.append(START_IDX['time_shift'] + finish_time)
                #Save it
                res.append(part)
                #Debug: comment when its done
                times.append(time - shift + finish_time)
                #Reset timer
                part = []
                time = 0
            else:
                part.append(event)
        else:
            #Velocity. Just append
            part.append(event)
    #The last part
    res.append(part)
    times.append(time)
    #print("Times: " + str(times))
    return res

def one_hot_vector(note):
    vector = np.zeros(vocab_size)
    vector[int(note)] = int(1)
    return vector

def one_hot_sequence(int_array):
    sequence = np.array([one_hot_vector(note) for note in int_array])
    sequence.reshape((len(int_array), vocab_size))
    print(sequence.shape)
    return sequence


#Will return the sequences X and Y correctly cut or padded, according to
#the maximum number of integers that we want
def crop_pad_sequences(seqX, seqY, pad_token, maxint=2048, sp_tokens=True):
    #seqX.insert(0, config.token_sos)
    #seqY.insert(0, config.token_sos)
    #seqX.append(config.token_eos)
    #seqY.append(config.token_eos)
    longestSeq = None
    shortestSeq = None
    x_longest = True
    
    if len(seqX)>=len(seqY):
        longestSeq = seqX
        shortestSeq = seqY
        x_longest = True
    else:
        longestSeq = seqY
        shortestSeq = seqX
        x_longest = False
        
    if len(longestSeq) >= maxint:
        longestSeq = longestSeq[0:maxint]
        #We have to calculate how much does the sequence with maxint ints lasts
        shortest_time = calculate_time(longestSeq)
        #print("Time of longest sequence cropped: " + str(shortest_time))
        slow_sequence = shortestSeq

        #Might be that the other sequence is longer than maxint ints too. Might even happen that,
        #between the maxint ints, the shortest sequence is actually faster.
        if len(shortestSeq) > maxint:
            shortestSeq = shortestSeq[0:maxint]
            slow_sequence = shortestSeq
            other_time = calculate_time(shortestSeq)
            #print("Time of shortest sequence cropped: " + str(other_time))
            if other_time < shortest_time:
                slow_sequence = longestSeq
                longestSeq = shortestSeq
                shortest_time = other_time
                x_longest = not x_longest
        
        slow_sequence = cut_time(slow_sequence, shortest_time)
        #print("Slow sequence cutted legnth: " + str(len(slow_sequence)))
        if sp_tokens:
            slow_sequence = pad_sequence(slow_sequence, pad_token, maxint)
        else:
            slow_sequence = pad_sequence(slow_sequence, pad_token, maxint)
        
        if sp_tokens:
            if x_longest:
                slow_sequence.insert(0, config.token_sos)
                slow_sequence = add_eos(slow_sequence)
            else:
                longestSeq.insert(0, config.token_sos)
                longestSeq = add_eos(longestSeq)
        
        if x_longest:
            return longestSeq, slow_sequence
        else:
            return slow_sequence, longestSeq
        
    else:
        x = pad_sequence(seqX, pad_token, maxint)
        if sp_tokens:
            y = pad_sequence(seqY, pad_token, maxint, add_eos=True)
            seqY.insert(0, config.token_sos)
        else:
            y = pad_sequence(seqY, pad_token, maxint)
        return x, y


r"""
Gets the xpos-th chunk of maxint length of x, gets the corresponding part of y,
and then cuts both as needed so that both have at max the maxint length.
Returns without padding.
"""
def get_sequences(x, y, maxint, xpos):
    xpos = maxint*xpos
    #Remove bad note offs
    seqX = get_correct_interval(x, xpos, maxint)
    #seqX = x[xpos:xpos+maxint]
    timeX = calculate_time(seqX)
    seqY = y
    #If the sequence is not the first one, we have to find where does the sequence
    #start in y
    if xpos > 0:
        prevX = x[0:xpos]
        prevTime = calculate_time(prevX)
        seqY = after_time_sequence(y, prevTime)
    
    seqY, timeY = cut_time_length(seqY, timeX, maxint)
    #Remove bad note offs (wont affect the time, but it might affect the length - too bad!)
    seqY = get_correct_interval(seqY, 0, len(seqY))
    if timeY < timeX:
        #We have to cut x to the time of Y
        seqX = cut_time(seqX, timeY)
        
    return seqX, seqY

r"""
Gets a subsequence starting at pos with the given size, but it makes sure that
no note_offs and note_ons are out of match (removes the bad ones)
Might return a smaller sequence if there are not enough events
"""
def get_correct_interval(seq, pos, size):
    note_on=[False]*RANGE_NOTE_ON
    i=0
    j=pos
    ret=[]
    while j < len(seq) and i < size:
        ev = seq[j]
        #Check note ons
        if ev < START_IDX['note_off']:
            if not note_on[ev]:
                #Correct note on
                ret.append(ev)
                #Reset
                note_on[ev] = True
                i+=1
                
        #Check note offs
        elif ev < START_IDX['time_shift']:
            if note_on[ev-128]:
                #Correct note off
                ret.append(ev)
                #Reset
                note_on[ev-128] = False
                i+=1
        #All other events are added
        else:
            ret.append(ev)
            i+=1
        j+=1
    
    return ret

#Calculates and returns how many timesteps are in the given sequence
def calculate_time(longSeq):
    longSeq_time = 0
    for event in longSeq:
        if (event >= START_IDX['time_shift']) and (event < START_IDX['velocity']):
            #Time event
            longSeq_time += event - START_IDX['time_shift']

    return longSeq_time


#Returns the sequence cut at a certain time
def cut_time(seq, maxtime):
    time=0
    for i, event in enumerate(seq):
        if (event >= START_IDX['time_shift']) and (event < START_IDX['velocity']):
            #Time event
            shift = event - START_IDX['time_shift']
            time += shift #Increment time
            if time > maxtime:
                #This last shift overlaps the splitting point
                finish_time = shift - (time - maxtime)
                ret = seq[0:i]
                ret.append(finish_time + START_IDX['time_shift'])
                return ret
    return seq

#Returns the sequence cut at the minimum between the specified time and the maximum length.
#Also returns the time that the returned sequence has
def cut_time_length(seq, maxtime, maxlength):
    if len(seq) <= maxlength:
        time = calculate_time(seq)
        return seq, time
    else:
        time=0
        finish_time=0
        i=0
        while (i < maxlength) and (time < maxtime):
            event = seq[i]
            if (event >= START_IDX['time_shift']) and (event < START_IDX['velocity']):
                #Time event
                shift = event - START_IDX['time_shift']
                time += shift #Increment time
                if time >= maxtime:
                    #This last shift overlaps the splitting point
                    finish_time = shift - (time - maxtime)
                    seq = seq[0:i]
                    seq.append(finish_time + START_IDX['time_shift'])
            i+=1
            
        if i == maxlength:
            seq = seq[0:maxlength]
        return seq, time

#Returns the subsequence that starts at the given time. It will modify the first time step to match exactly the time.
#It can return an empty list if the given time is larger than the time of the sequence.
def after_time_sequence(seq, maxtime):
    i=0
    time=0
    while (i < len(seq)) and (time < maxtime):
        event = seq[i]
        if (event >= START_IDX['time_shift']) and (event < START_IDX['velocity']):
            #Time event
            shift = event - START_IDX['time_shift']
            time += shift #Increment time
            if time >= maxtime:
                #This last shift overlaps the starting point
                start_time = time - maxtime #The excess of maxtime
                ret = seq[i:]
                ret[0] = start_time + START_IDX['time_shift']
                return ret
        i+=1

    return []

#Cuts the sequence at maxint-1 and adds the end of sequence token
def cut_int(seq, maxint):
    ret = seq[0:(maxint-1)]
    ret.append(config.token_eos)
    return ret

#Input: a shorter sequence than maxint.
#Return: a maxint sequence padded with the given token
def pad_sequence(seq, pad_token, maxint=2048, add_eos=False):
    if add_eos:
        seq.append(config.token_eos)
        for i in range(maxint-len(seq)+1):
            seq.append(pad_token)
    else:
        for i in range(maxint-len(seq)):
            seq.append(pad_token)
    return seq

#Adds the eos token where it's needed in a padded sequence. The resulting array
#has one more item
def add_eos(seq):
    i = len(seq)
    while i> 0 and seq[i-1] == 388:
        i = i - 1
    seq.insert(i, config.token_eos)
    return seq


r""" Splits lst in chunks of size length and returns them in a list. If the last
chunk has less than min_length, it is ommited.
"""
def split_even_list(lst, size, min_length):
    array = [lst[i:min(len(lst), i+size)] for i in range(0, len(lst), size)]
    if len(array[-1]) < min_length:
        array.pop()
    return array

r"""
Returns the sequence without the velocity events
"""
def remove_velocity(seq):
    return [ev for ev in seq if ev < START_IDX['velocity']]

r"""
Checks the causal relationship note on -> note off
"""
def check_validity(seq):
    note_on=[False]*RANGE_NOTE_ON
    i=0
    while i < len(seq):
        ev = seq[i]
        #Check note ons
        if ev < START_IDX['note_off']:
            if note_on[ev]:
                print("Double note on at {}: {}".format(i, ev))
                return False
            else:
                note_on[ev] = True
                
        #Check note offs
        elif ev < START_IDX['time_shift']:
            if not note_on[ev-128]:
                print("Bad note off at {}: {}".format(i, ev))
                return False
            else:
                note_on[ev-128] = False
        i+=1
    return True
    