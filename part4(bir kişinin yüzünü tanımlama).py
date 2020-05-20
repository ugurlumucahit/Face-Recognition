import cv2

def generate_dataset(img,id,img_id):
    cv2.imwrite("data2/user."+str(id)+"."+str(img_id)+".jpg",img)  #resimleri kaydedecegimiz yer
# id=>kullanıcı Id'si
#img_id => resimler ne kadar fazlaysa, classifier da o kadar basarılı olur


    
def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
#yüz etrafına sınır getiren fonksiyon// ilk parametre yüzü bulmak istedigimiz resim
#göz,burun,ağız için kullanacagımız sınıflandırıcı


#cercevenin rengi
#tanımladıgı objenin yanında ne oldugunu belirtecek bir yazı
    gray_img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #yüz,göz,burun,ağız gibi materyalleri resmi griye cevirdikten sonra bulacagız.
    features =classifier.detectMultiScale(gray_img,scaleFactor,minNeighbors) #classifier içinde materyal)yüz,ağız vb) tanımlama//detectMultiScale => classifier da ki materyal ne ise, resimdeki bütün olası materyalleri tanımlar(örn:Yüz)
    coords =[]
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8, color,1,cv2.LINE_AA)
        coords =[x,y,w,h]
    return coords



def detect(img,faceCascade,eyeCascade,noseCascade,mouthCascade,img_id):
    color ={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0),"white":(255,255,255)}
    coords =draw_boundary(img,faceCascade,1.1,10,color["blue"],"Face")

    if len(coords) ==4:  #coords(x,y,h,w)
        roi_img = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]] #yüz tanımlama resmindeki işlem #sadece yüze odaklanma işlemi

        user_id =1
        generate_dataset(roi_img,user_id,img_id)

        #coords = draw_boundary(roi_img, eyeCascade, 1.1, 13, color["red"], "Eyes")
        #coords = draw_boundary(roi_img, noseCascade, 1.1, 5, color["green"], "Nose")
        #coords = draw_boundary(roi_img, mouthCascade, 1.1, 20, color["white"], "Mouth")
    return img

faceCascade =cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #yüz için
eyeCascade =cv2.CascadeClassifier("haarcascade_eye.xml") #göz için
noseCascade =cv2.CascadeClassifier("haarcascade_mcs_nose.xml") #burun için
mouthCascade =cv2.CascadeClassifier("haarcascade_smile.xml") #ağız için


#scaleFactor => classifier materyali tanımladıgında, o materyali ölceklendirir
#minNeighbors => materyali tanımlamadan önce ki aşama..//1,2,3 // diyelim ki 3.. 3 classifier da onaylar ise o zaman bu bir yüz dür, diye tanımlar.
video_capture =cv2.VideoCapture(0)  #Webcam den video akışı için gerekli( 0 pc nin webcami/1 veya -1 ise 3.parti kameralar)


img_id =0

while True:
    _,img =video_capture.read()    #videoyu resim olarak okuyacagız// video_capture.read() 2 parametre döndürür, // ilk parametreyi kullanmayacagımızdan dolayı o sekilde tanımladık
    img =detect(img,faceCascade,eyeCascade,noseCascade,mouthCascade,img_id)
    cv2.imshow("face detection",img)   #ilk parametre / pencere adı , ikinci parametre / okumak,bastırmak istedigimiz resim,video vb.
    img_id +=1
    if cv2.waitKey(1) & 0xFF == ord("q"): #döngüden cıkacagız eğer ki q'ya basarsak. // waitkey=> tusa basma için bekleme metodu içine koydugumuz deger ise milisecond cinsinden/ 0 koyarsak sonsuza kadar bekler taa ki q ya basana kadar

        break
video_capture.release()  #webcam'i serbest bırakma
cv2.destroyAllWindows()