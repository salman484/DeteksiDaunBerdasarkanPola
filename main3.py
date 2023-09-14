import sys
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QMainWindow, QFileDialog, QApplication
from PyQt5.uic import loadUi
import numpy as np
import glob
import imutils



class showImage(QMainWindow):
    def __init__(self):
        super(showImage, self).__init__()
        loadUi('untitled.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.ProsesButton.clicked.connect(self.ProsesClicked)


    # membuat prosedur button load click
    @pyqtSlot()
    def loadClicked(self):
        flname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\\MSI\\PycharmProjects\\ProjekAkhir\\contoh gambar',
                                                     "Image Files(*.jpg)")
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

    def loadImage(self, flname): #C:\Users\MSI\PycharmProjects\ProjekAkhir
        self.image = cv2.imread(flname)
        self.displayImage()

    def ProsesClicked(self):
        # menyimpan data template
        template_data = []
        files1 = glob.glob('template/*.jpg')

        # Convert Greyscale
        for myfile in files1:
            image = cv2.imread(myfile)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = 100
            h, w = image.shape[:2]
            for i in np.arange(h):
                for j in np.arange(w):
                    a = image.item(i, j)
                    b = a + brightness
                    if b > 255:
                        b = 255
                    elif b < 0:
                        b = 0
                    else:
                        b = b
                    image.itemset((i, j), b)

            delta = 0
            scale = 1
            ddepth = cv2.CV_16S

            # Gradient-X
            grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            # grad_x = cv2.Scharr(gray,ddepth,1,0)
            # Gradient-Y
            grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            # grad_y = cv2.Scharr(gray,ddepth,0,1)

            abs_grad_x = cv2.convertScaleAbs(grad_x)  # converting back to uint8
            abs_grad_y = cv2.convertScaleAbs(grad_y)

            dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            template_data.append(dst)

            # Perulangan untuk template matching
            for tmp in template_data:
                (tM, tW) = tmp.shape[:2]
                cv2.imshow("Template", tmp)

                # Melakukan Perulangan untuk meload data gambar
                for imageP in glob.glob('Gambar/*.jpg'):
                    # Convert Greyscale
                    imageS = cv2.imread(imageP)
                    gray = cv2.cvtColor(imageS, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('Grayscale', gray)
                    found = None
                    # Melakukan Perngulangan untuk scaling gambar
                    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                        # Merescale gambar sesuai dengan skala dan rasio yang diberikan
                        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
                        r = gray.shape[1] / float(resized.shape[1])

                        # Jika gambar yang di rescale lebih kecil dari template
                        # break loop
                        if resized.shape[0] < tM or resized.shape[1] < tW:
                            break

                        # Deteksi Tepi untuk gambar yang telah discale
                        # Melakukan Template Matching
                        delta = 0
                        scale = 1
                        ddepth = cv2.CV_16S

                        # Gradient-X
                        grad_x = cv2.Sobel(resized, ddepth, 1, 0, ksize=3, scale=scale, delta=delta,
                                           borderType=cv2.BORDER_DEFAULT)
                        # grad_x = cv2.Scharr(gray,ddepth,1,0)
                        # Gradient-Y
                        grad_y = cv2.Sobel(resized, ddepth, 0, 1, ksize=3, scale=scale, delta=delta,
                                           borderType=cv2.BORDER_DEFAULT)
                        # grad_y = cv2.Scharr(gray,ddepth,0,1)

                        abs_grad_x = cv2.convertScaleAbs(grad_x)  # converting back to uint8
                        abs_grad_y = cv2.convertScaleAbs(grad_y)

                        dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

                        template_data.append(dst)
                        cv2.imshow("Hasil", dst)


                        result = cv2.matchTemplate(dst, tmp, cv2.TM_CCOEFF_NORMED)
                        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                        # Jika ketemu dengan variable berskala baru, maka simpan
                        if found is None or maxVal > found[0]:
                            found = (maxVal, maxLoc, r)
                            if maxVal >= 0.4:
                                rep = ""
                                imageP1 = imageP
                                imageP1 = imageP1.replace(rep, "")
                                imageP1 = imageP1.replace("\\", " ")
                                imageP1 = imageP1.replace(".jpg", "")
                                print("ini adalah", imageP1, "")
                    # Keluarkan variable gambar yang telah di skala dan kembalikan ke skala
                    # berikan kotakan pada gambar asli jika ditemukan kecocokan template
                    (maxVal, maxLoc, r) = found
                    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tM) * r))
                    if maxVal >= 0.4:
                        cv2.rectangle(imageS, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.imshow("Image", imageS)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888

            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()

        if windows == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)
        if windows == 2:
            self.img2Label.setPixmap(QPixmap.fromImage(img))
            self.img2Label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.img2Label.setScaledContents(True)
            return self.imgLabel

        self.imgLabel.setPixmap(QPixmap.fromImage(img))
        self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def save_Button(self):
        flname, filter = QFileDialog.getSaveFileName(self, 'save file', 'D:\\',
                                                     "Images Files(*.jpg)")
        if flname:
            cv2.imwrite(flname, self.image)
        else:
            print('Saved')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = showImage()
    window.setWindowTitle('Show Image GUI')
    window.show()
    sys.exit(app.exec_())

