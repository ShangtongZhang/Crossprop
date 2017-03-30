prefix = '../Crossprop/train'
# templatePY = 'templateYAD.py'
templatePY = 'template.py'
templateSH = 'template.sh'

appendix = [
    '''
train(stepSizes[0], 500, 6500)
    ''',
    '''
train(stepSizes[1], 500, 6500)
    ''',
    '''
train(stepSizes[2], 500, 6500)
    ''',
    '''
train(stepSizes[3], 500, 6500)
    ''',
    '''
train(stepSizes[4], 500, 6500)
    ''',
    '''
train(stepSizes[5], 500, 6500)
    ''',
    '''
train(stepSizes[0], 100, 6500)
train(stepSizes[1], 100, 6500)
    ''',
    '''
train(stepSizes[2], 100, 6500)
train(stepSizes[3], 100, 6500)
    ''',
    '''
train(stepSizes[4], 100, 6500)
train(stepSizes[5], 100, 6500)
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
