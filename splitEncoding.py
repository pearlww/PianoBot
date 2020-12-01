
import numpy as np

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

vocab_size = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT

#max_time in milliseconds
def split_encoding(int_array, max_time):
    max_time = max_time / 10 #In time shift units
    time = 0
    note_array=[False]*RANGE_NOTE_ON  #Note on/Note off array
    res=[]
    part=[]
    
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
    return res

def one_hot_vector(note):
    vector = np.zeros(vocab_size)
    vector[int(note)] = 1
    return vector

def one_hot_sequence(int_array):
    sequence = np.array([one_hot_vector(note) for note in int_array])
    sequence.reshape((len(int_array), vocab_size))
    print(sequence.shape)
    return sequence
