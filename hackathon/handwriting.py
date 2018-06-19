## import mnist tester file
from trainingStuff.MNISTTester import MNISTTester
import cv2
from Tkinter import *
import imutils
## import training data and neural network
data_path = 'trainingStuff/mnist/data/'
model_path = 'trainingStuff/models/mnist-cnn'

mnist = MNISTTester(model_path=model_path,data_path=data_path)

f = False
t = True


# ======================= Main Variables ========================== #

takePicture = True

## 2 = get answers
checkType = 3
## 3 = check answers

file = "1234.jpg"
showImage = True

# ================================================================= #

if takePicture:
	cap = cv2.VideoCapture(0)
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	while True:
		ret, frame = cap.read()
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
		#cropped = rgb[0:620, 0:440]


		cv2.imshow('frame', cv2.flip(rgb,1)) ## flips frame
		cv2.resizeWindow('frame',600,400)

		if cv2.waitKey(1) & 0xFF == ord('c'): ## press 'c' to take picture
			img = cv2.imwrite('imgs/scanned.jpg', frame)
			break

	cap.release()
	cv2.destroyAllWindows()

	im = mnist.splitImage("imgs/scanned.jpg",showImage=showImage)
	rects = im["rects"]
	img = im["numbers"]

else:
	try:
		im = mnist.splitImage("imgs/"+file,showImage=showImage)
		rects = im["rects"]
		img = im["numbers"]
	except Exception:
		raise Exception("Could not find image. Make sure image is in 'imgs' folder")




##############################################################################################################
nums = []
## PREDICT NUMBERS
for item in img:
	accuracy = mnist.predict(item)["accuracy"]
	#if accuracy >= 0.9:
	#print accuracy
	num = mnist.predict(item)["number"]
	nums.append(num)
#print nums 
## COMBINE DOUBLE DIGITS ######################################################

maxx = 0
maxy = 0
for i in rects:
  if i[1] > maxy:
	maxy = i[1]

for i in rects:
  if i[0] > maxx:
	maxx = i[0]

y_ratio = int(maxy * 0.3)
x_ratio = int(maxx * 0.3)

nums1 = []
skip = False
for i in range(len(rects)-1):
	## if double digit
	if skip == True:
		if i == len(nums) - 2: ## -2 because -1 is not inclusive for the range
			skip = True
			## nums1.append(nums[i])
			continue
		skip = False
	elif (i < len(nums)-1) and (((rects[i+1][0]-rects[i][0]) in range(0, x_ratio)) and ((rects[i+1][1]-rects[i][1]) in range(0, y_ratio) or (rects[i][1] - rects[i+1][1]) in range(0, y_ratio)) ):      
			if skip == True:
				skip = False
				continue
			nums1.append((nums[i])*10 + nums[i+1])
			skip = True
	## if single digit
	else:
		if i == len(nums) - 2: ## -2 because -1 is not inclusive for the range
			skip = True
		nums1.append(nums[i])
	

## account for last element if previous was not double digit 
## after all of the double digit classification
if skip == True:
	nums1.append(nums[-1])

nums = nums1


## SEPARATE NUMBERS INTO PROBLEMS
## factor = len(nums)/checkType

nums = [nums[x:x+checkType] for x in range(0, len(nums),checkType)]

if len(nums[-1]) % checkType != 0:
	nums.pop(-1)


## CHECK/GET ANSWERS
right = 0
wrong = 0
answers =[]
wrongAnsws = []
for i in range(len(nums)):
	if checkType == 3:
		if nums[i][0] + nums[i][1] == nums[i][2]:
			right +=1
		else:
			wrong +=1
			wrongAnsws.append(nums.index(nums[i])+1)
	answer = nums[i][0] + nums[i][1]
	answers.append(answer)


for item in nums:
	if checkType == 2:
		item.insert(1,"+")
	elif checkType == 3:
		item.insert(1,"+")
		item.insert(3,"=")
	item = list(item)
	"".join([str(n) for n in item])
print nums ## after sorting into problems


if checkType == 2:
	print "Answers: {}".format(answers)

elif checkType == 3:
	print "{}% correct".format((float(right)/float(wrong+right))*100)
	print "{}/{} correct".format(right,right+wrong)
	if wrongAnsws != []:
		print "Wrong answers: {}".format(wrongAnsws)
		print "Correct Answers: {}".format(answers)

