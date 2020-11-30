
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

#max_time in milliseconds
def split_encoding(int_array, max_time):
    max_time = max_time / 8 #In time shift units
    time = 0
    note_array=[False]*128  #Note on/Note off array
    res=[]
    part=[]
    
    for event in int_array:
        if event < START_IDX['note_off']:
            #note_on event
            note_array[event] = True
            part.append(event)
        elif event < START_IDX['time_shift']:
            #note off event
            if note_array[event - START_IDX['note_off']]:
                note_array[event] = False
                part.append(event)
            #Note that, in the else clause, we keep the note off 
            #AND we don't add that unnecessary event to the part
            #Can happen after splits
        elif event < START_IDX['velocity']:
            #time shift event
            time += event - START_IDX['time_shift'] #Increment time
            if time >= max_time:
                #This last shift overlaps the splitting point
                finish_time = time - max_time
                #Append the correct duration to the part
                part.append(START_IDX['time_shift'] + finish_time)
                #Save it
                res.append(part)
                #Reset timer
                time = 0
            else:
                part.append(event)
        else:
            #Velocity. Just append
            part.append(event)
    
    return res
