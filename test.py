baseDir = '/Users/Sammy/repos/CSMC_35200_BraintoTextGroup/NeuralDecoder'
import nltk
nltk.download('averaged_perceptron_tagger_eng')
#import os
#os.makedirs(baseDir+'/competitionData', exist_ok=True)

from makeTFRecordsFromSession import makeTFRecordsFromCompetitionFiles 
from getSpeechSessionBlocks import getSpeechSessionBlocks
blockLists = getSpeechSessionBlocks()

for sessIdx in range(len(blockLists)):
    sessionName = blockLists[sessIdx][0]
    dataPath = baseDir + '/data/competitionData'
    tfRecordFolder = baseDir + '/derived/tfRecords/'+sessionName
    makeTFRecordsFromCompetitionFiles(sessionName, dataPath, tfRecordFolder)