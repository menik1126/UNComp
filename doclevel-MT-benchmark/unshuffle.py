#! /usr/bin/env python3


import sys, collections

shuffledRef = sys.argv[1]
unshuffledRef = sys.argv[2]
shuffledSys = sys.argv[3]
unshuffledSys = sys.argv[4]

print("  Infer sentence correspondences:", shuffledRef, "<=>", unshuffledRef)
shuffledCorr = collections.defaultdict(list)
i = 0
for line in open(shuffledRef, 'r', encoding='utf-8'):
	if line.isspace():
		continue
	shuffledCorr[line.strip()].append(i)
	i += 1
print("    Shuffled lines:       ", i+1)
print("    Unique shuffled lines:", len(shuffledCorr))

shuf2unshuf = {}
i = 0
for line in open(unshuffledRef, 'r', encoding='utf-8'):
	if line.isspace():
		continue
	if (line.strip() in shuffledCorr) and (shuffledCorr[line.strip()] != []):
		j = shuffledCorr[line.strip()].pop()
		shuf2unshuf[j] = i
	else:
		print("    Sentence not found:", line.strip())
	i += 1
print("    Unshuffled lines:     ", i+1)
print("    Correspondences found:", len(shuf2unshuf))
#print(shuf2unshuf)

print("  Unshuffle file:", shuffledSys, "=>", unshuffledSys)
data = {}
for i, line in enumerate(open(shuffledSys, 'r', encoding='utf-8')):
	data[shuf2unshuf[i]] = line

w = open(unshuffledSys, 'w', encoding='utf-8')
for i in sorted(data):
	w.write(data[i])
w.close()
print("  Done")
