prefix = '../Crossprop/train'
# templatePY = 'templateYAD.py'
templatePY = 'template.py'
templateSH = 'template.sh'

# appendix = [
#     '''
# for step in stepSizes[:2]:
#     train(step, 200, 40500)
#     ''',
#     '''
# for step in stepSizes[2: 4]:
#     train(step, 200, 40500)
#     ''',
#     '''
# for step in stepSizes[4:]:
#     train(step, 200, 40500)
#     ''',
#     '''
# for step in stepSizes[:2]:
#     train(step, 200, 23500)
#     ''',
#     '''
# for step in stepSizes[2: 4]:
#     train(step, 200, 23500)
#     ''',
#     '''
# for step in stepSizes[4:]:
#     train(step, 200, 23500)
#     ''',
#     '''
# for step in stepSizes[:2]:
#     train(step, 200, 18500)
#     train(step, 200, 13500)
#     ''',
#     '''
# for step in stepSizes[2: 4]:
#     train(step, 200, 18500)
#     train(step, 200, 13500)
#     ''',
#     '''
# for step in stepSizes[4:]:
#     train(step, 200, 18500)
#     train(step, 200, 13500)
#     '''
# ]
#
# appendix = [
#     '''
# for step in stepSizes[:2]:
#     train(step, 60, 3500)
#     ''',
#     '''
# for step in stepSizes[2: 4]:
#     train(step, 60, 3500)
#     ''',
#     '''
# for step in stepSizes[4:]:
#     train(step, 60, 3500)
#     ''',
#     '''
# for step in stepSizes[:2]:
#     train(step, 60, 6500)
#     ''',
#     '''
# for step in stepSizes[2: 4]:
#     train(step, 60, 6500)
#     ''',
#     '''
# for step in stepSizes[4:]:
#     train(step, 60, 6500)
#     ''',
#     '''
# for step in stepSizes[:2]:
#     train(step, 60, 9500)
#     ''',
#     '''
# for step in stepSizes[2: 4]:
#     train(step, 60, 9500)
#     ''',
#     '''
# for step in stepSizes[4:]:
#     train(step, 60, 9500)
#     '''
# ]

appendix = [
    '''
train(stepSizes[0], 900, 24500)
    ''',
    '''
train(stepSizes[1], 900, 24500)
    ''',
    '''
train(stepSizes[2], 900, 24500)
    ''',
    '''
train(stepSizes[3], 900, 24500)
    ''',
    '''
train(stepSizes[4], 900, 24500)
    ''',
    '''
train(stepSizes[5], 900, 24500)
    ''',
    '''
for step in stepSizes[:3]:
    train(step, 900, 15500)
    ''',
    '''
for step in stepSizes[3:]:
    train(step, 900, 15500)
    ''',
    '''
for step in stepSizes:
    train(step, 900, 3500)
    train(step, 900, 6500)
    '''
]

appendix = [
    '''
for sample in samples:
    train(stepSizes[0], 500, sample)
    ''',
    '''
for sample in samples:
    train(stepSizes[1], 500, sample)
    ''',
    '''
for sample in samples:
    train(stepSizes[2], 500, sample)
    ''',
    '''
for sample in samples:
    train(stepSizes[3], 500, sample)
    ''',
    '''
for sample in samples:
    train(stepSizes[4], 500, sample)
    ''',
    '''
for sample in samples:
    train(stepSizes[5], 500, sample)
    ''',
    '''
for sample in samples:
    train(stepSizes[0], 100, sample)
    train(stepSizes[1], 100, sample)
    ''',
    '''
for sample in samples:
    train(stepSizes[2], 100, sample)
    train(stepSizes[3], 100, sample)
    ''',
    '''
for sample in samples:
    train(stepSizes[4], 100, sample)
    train(stepSizes[5], 100, sample)
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
