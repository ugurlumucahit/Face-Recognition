import cv2

video_capture =cv2.VideoCapture(0)  #Webcam den video akışı için gerekli( 0 pc nin webcami/1 veya -1 ise 3.parti kameralar)

while True:
    _,img =video_capture.read()    #videoyu resim olarak okuyacagız// video_capture.read() 2 parametre döndürür, // ilk parametreyi kullanmayacagımızdan dolayı o sekilde tanımladık
    cv2.imshow("face detection",img)   #ilk parametre / pencere adı , ikinci parametre / okumak,bastırmak istedigimiz resim,video vb.
    if cv2.waitKey(1) & 0xFF == ord("q"): #döngüden cıkacagız eğer ki q'ya basarsak. // waitkey=> tusa basma için bekleme metodu içine koydugumuz deger ise milisecond cinsinden/ 0 koyarsak sonsuza kadar bekler taa ki q ya basana kadar

        break
video_capture.release()  #webcam'i serbest bırakma
cv2.destroyAllWindows()