prefix = '../Crossprop/train'
templatePY = 'templateYAD.py'
templateSH = 'template.sh'

appendix = [
    '''
for step in stepSizes[:3]:
    train(step, hiddenUnits[0], samples[0])
    ''',
    '''
for step in stepSizes[3:]:
    train(step, hiddenUnits[0], samples[0])
    ''',
    '''
for step in stepSizes[:2]:
    train(step, hiddenUnits[0], samples[1])
    ''',
    '''
for step in stepSizes[2: 5]:
    train(step, hiddenUnits[0], samples[1])
    ''',
    '''
for step in stepSizes[5:]:
    train(step, hiddenUnits[0], samples[1])
    ''',
    '''
for step in stepSizes[:2]:
    train(step, hiddenUnits[0], samples[2])
    ''',
    '''
for step in stepSizes[2: 5]:
    train(step, hiddenUnits[0], samples[2])
    ''',
    '''
for step in stepSizes[5:]:
    train(step, hiddenUnits[0], samples[2])
    '''
]

appendix = [
    '''
train(stepSizes[0], hiddenUnits[0], samples[0])
    ''',
    '''
train(stepSizes[1], hiddenUnits[0], samples[0])
    ''',
    '''
train(stepSizes[2], hiddenUnits[0], samples[0])
    ''',
    '''
train(stepSizes[3], hiddenUnits[0], samples[0])
    ''',
    '''
train(stepSizes[4], hiddenUnits[0], samples[0])
    ''',
    '''
train(stepSizes[5], hiddenUnits[0], samples[0])
    ''',
    '''
train(stepSizes[6], hiddenUnits[0], samples[0])
    '''
]

step = 3
factor = 10

for appendixInd, appendix in enumerate(appendix):
    for fileIndex in range(0, factor):
        fr = open(templatePY, 'r')
        fw = open(prefix + str(appendixInd * factor + fileIndex + 1) + '.py', 'w')
        for line in fr.readlines():
            if line.find('@@@') >= 0:
                line = line.replace('@@@', str((fileIndex + 1) * step))
            elif line.find('@@') >= 0:
                line = line.replace('@@', str(fileIndex * step))
            fw.write(line)
        fw.write(appendix)
        fw.close()
        fr.close()

        fr = open(templateSH)
        fw = open(prefix + str(appendixInd * factor + fileIndex + 1) + '.sh', 'w')
        for line in fr.readlines():
            if line.find('@@') >= 0:
                line = line.replace('@@', str(appendixInd * factor + fileIndex + 1))
            fw.write(line)
        fw.close()
        fr.close()
