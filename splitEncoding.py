
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
def crop_pad_sequences(seqX, seqY, pad_token, maxint=2048):
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
        
    if len(longestSeq) > maxint:
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
        slow_sequence = pad_sequence(slow_sequence, pad_token, maxint)
        
        if x_longest:
            return longestSeq, slow_sequence
        else:
            return slow_sequence, longestSeq
        
    else:
        x = pad_sequence(seqX, pad_token, maxint)
        y = pad_sequence(seqY, pad_token, maxint)
        return x, y

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
                ret.append(finish_time)
                return ret
    return seq

#Input: a shorter sequence than maxint.
#Return: a maxint sequence padded with the given token
def pad_sequence(seq, pad_token, maxint=2048):
    for i in range(maxint-len(seq)):
        seq.append(pad_token)
    return seq
