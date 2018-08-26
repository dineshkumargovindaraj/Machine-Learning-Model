import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# load the model from disk
loaded_model = pickle.load(open("C:\\Users\\dkdin\\Desktop\\DataMining\\DataMining\\Assignment1\\trained_model.sav", 'rb'))

# Load the real data from here

real_data = ["From: Mamatha Devineni Ratnam <mr47+@andrew.cmu.edu>\
                Subject: Pens fans reactions\
                Organization: Post Office, Carnegie Mellon, Pittsburgh, PA\
                Lines: 12\
                NNTP-Posting-Host: po4.andrew.cmu.edu\
\
\
\
                I am sure some bashers of Pens fans are pretty confused about the lack\
                of any kind of posts about the recent Pens massacre of the Devils. Actually,\
                I am  bit puzzled too and a bit relieved. However, I am going to put an end\
                to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they\
                are killing those Devils worse than I thought. Jagr just showed you why\
                he is much better than his regular season stats. He is also a lot\
                fo fun to watch in the playoffs. Bowman should let JAgr have a lot of\
                fun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final\
                regular season game.          PENS RULE!!!" ,
             
             
             "From: Alexander Samuel McDiarmid <am2o+@andrew.cmu.edu>\
            Subject: driver ??\
            Organization: Sophomore, Mechanical Engineering, Carnegie Mellon, Pittsburgh, PA\
            Lines: 15\
            NNTP-Posting-Host: po4.andrew.cmu.edu\
\
\
            1)    I have an old Jasmine drive which I cannot use with my new system.\
             My understanding is that I have to upsate the driver with a more modern\
            one in order to gain compatability with system 7.0.1.  does anyone know\
            of an inexpensive program to do this?  ( I have seen formatters for <$20\
            buit have no idea if they will work)\
\
            2)     I have another ancient device, this one a tape drive for which\
            the back utility freezes the system if I try to use it.  THe drive is a\
            jasmine direct tape (bought used for $150 w/ 6 tapes, techmar\
            mechanism).  Essentially I have the same question as above, anyone know\
            of an inexpensive beckup utility I can use with system 7.0.1\
\
            all help and advice appriciated."

                       
            
            ]

expected = [10,4]

prediction = loaded_model.predict(real_data)
print(accuracy_score(expected, prediction))
print(confusion_matrix(expected, prediction))
print(classification_report(expected, prediction))