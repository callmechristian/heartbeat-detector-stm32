from pydub import AudioSegment
import os
import uuid

t1 = 0
t2 = 1000

leng = 0

classes = ["Wednesday", "Purple", "Sunrise"]

# pydub doesn't support loading more than one wav ? or idk how to do it
j = 0

leng = len(AudioSegment.from_wav(classes[j] + ".wav"))
print(int(leng/1000))
#if it doesn’t exist we create a folder
if not os.path.exists(classes[j]):
    os.makedirs(classes[j])

for i in range(0, int(leng/1000)):
    mkid = uuid.uuid1()
    newAudio = AudioSegment.from_wav(classes[j] + ".wav")
    newAudio = newAudio[t1:t2]
    newAudio.export(classes[j] + "/" + str(i) + "_" + mkid + "_" + classes[j] + ".wav", format="wav") #Exports to a wav file in the current path.
    t1 = t1 + 1000 #Works in milliseconds
    t2 = t2 + 1000
    print("Writing " + str(i) + "_" + mkid + "_" + classes[j] + ".wav\n")