prefix = '../Crossprop/train'
templatePY = 'template.py'
templateSH = 'template.sh'

step = 3
for fileIndex in range(0, 10):
    fr = open(templatePY, 'r')
    fw = open(prefix + str(fileIndex + 1) + '.py', 'w')
    for line in fr.readlines():
        if line.find('@@@') >= 0:
            line = line.replace('@@@', str((fileIndex + 1) * step))
        elif line.find('@@') >= 0:
            line = line.replace('@@', str(fileIndex * step))
        fw.write(line)
    fw.close()
    fr.close()

    fr = open(templateSH)
    fw = open(prefix + str(fileIndex + 1) + '.sh', 'w')
    for line in fr.readlines():
        if line.find('@@') >= 0:
            line = line.replace('@@', str(fileIndex + 1))
        fw.write(line)
    fw.close()
    fr.close()
