from pydub import AudioSegment
import os
import uuid
import math

t1 = 0
t2 = 1000

leng = 0

classes = ["Purple", "Sunrise", "Wednesday"]
mods = ["higher-1", "higher-2", "lower-1", "lower-2", "reverb", "distortion"]

ctr_files = 0

for j in range(len(mods)):
    for k in range(len(classes)):
        file_name = "Samples-" + mods[j] + "-0" + str(k+1) + ".wav"
        # FIXED: subsequent file loads blank -> due to file handler in pydub not properly (or quickly enough) closing
        with open(file_name, "rb") as f:
            audio_data = f.read()
            audio = AudioSegment.from_wav(file_name)
            leng = math.ceil(len(audio)/1000)

        #if it doesn’t exist we create a folder
        if not os.path.exists(classes[k]):
            os.makedirs(classes[k])

        # reset times
        t1 = 0
        t2 = 1000

        for i in range(0, leng):
            mkid = str(uuid.uuid1())
            newAudio = audio[t1:t2]
            # newAudio.export(classes[k] + "/" + mkid + "_" + mods[j] + "_" + classes[k] + ".wav", format="wav") #Exports to a wav file
            t1 = t1 + 1000 #Works in milliseconds
            t2 = t2 + 1000
            print("Wrote: " + mkid + "_" + mods[j] + "_" + classes[k] + ".wav")
            ctr_files += 1

print("Created " + str(ctr_files) + " new audio samples. Boom. :)")