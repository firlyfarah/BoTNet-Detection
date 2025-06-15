import numpy as np
import random as rnd


class SVM():
    def __init__(self, max_iter=10000, C=1.0, epsilon=0.001):
        self.kernel = self.kernel_linear
        self.max_iter = max_iter
        self.C = C
        self.epsilon = epsilon

    def fit(self, fitur, label):
        jumlah_data = fitur.shape[0]
        nilai_alpha = np.zeros((jumlah_data))
        iterasi = 0

        self.matriks_kernel = np.zeros((jumlah_data, jumlah_data))
        for i in range(jumlah_data):
            for j in range(jumlah_data):
                self.matriks_kernel[i, j] = self.kernel(fitur[i], fitur[j])

        while True:
            iterasi += 1
            alpha_sebelumnya = np.copy(nilai_alpha)

            for j in range(jumlah_data):
                i = self.get_rnd_int(0, jumlah_data - 1, j)
                if i == j:
                    continue

                fitur_i = fitur[i, :]
                fitur_j = fitur[j, :]
                label_i = label[i]
                label_j = label[j]

                kernel_ij = self.matriks_kernel[i, i] + self.matriks_kernel[j, j] - 2 * self.matriks_kernel[i, j]
                if kernel_ij == 0:
                    continue

                alpha_j_awal, alpha_i_awal = nilai_alpha[j], nilai_alpha[i]
                (L, H) = self.hitung_batas_L_H(self.C, alpha_j_awal, alpha_i_awal, label_j, label_i)

                error_i = self.hitung_error(i, label_i, nilai_alpha, label, self.matriks_kernel, self.bias if hasattr(self, 'bias') else 0)
                error_j = self.hitung_error(j, label_j, nilai_alpha, label, self.matriks_kernel, self.bias if hasattr(self, 'bias') else 0)

                nilai_alpha[j] = alpha_j_awal + float(label_j * (error_i - error_j)) / kernel_ij
                nilai_alpha[j] = max(nilai_alpha[j], L)
                nilai_alpha[j] = min(nilai_alpha[j], H)
                nilai_alpha[i] = alpha_i_awal + label_i * label_j * (alpha_j_awal - nilai_alpha[j])

            perubahan = np.linalg.norm(nilai_alpha - alpha_sebelumnya)
            if perubahan < self.epsilon:
                break

            if iterasi >= self.max_iter:
                print("Jumlah iterasi melebihi batas maksimum:", self.max_iter)
                break

        self.nilai_alpha = nilai_alpha
        self.vektor_pendukung = np.where(nilai_alpha > 1e-5)[0]
        self.bias = self.hitung_bias(label, nilai_alpha, self.matriks_kernel)
        self.bobot = self.hitung_bobot(nilai_alpha, label, fitur)

    def predict(self, fitur_baru):
        return self.fungsi_keputusan(fitur_baru, self.bobot, self.bias)

    def hitung_bias(self, label, nilai_alpha, matriks_kernel):
        b = 0
        for i in range(len(label)):
            b += label[i] - np.sum(nilai_alpha * label * matriks_kernel[:, i])
        return b / len(label)

    def hitung_bobot(self, nilai_alpha, label, fitur):
        return np.dot(nilai_alpha * label, fitur)

    def fungsi_keputusan(self, fitur, bobot, bias):
        hasil = np.dot(fitur, bobot.T) + bias
        return np.where(hasil >= 0, 1, -1)

    def hitung_error(self, index, label_ke, nilai_alpha, label, matriks_kernel, bias):
        fx = np.sum(nilai_alpha * label * matriks_kernel[:, index]) + bias
        return fx - label_ke

    def hitung_batas_L_H(self, C, alpha_j, alpha_i, label_j, label_i):
        if label_i != label_j:
            return (max(0, alpha_j - alpha_i), min(C, C - alpha_i + alpha_j))
        else:
            return (max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j))

    def get_rnd_int(self, a, b, selain_ini):
        hasil = selain_ini
        percobaan = 0
        while hasil == selain_ini and percobaan < 1000:
            hasil = rnd.randint(a, b)
            percobaan += 1
        return hasil

    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def print_info(self):
        print("Parameter Algoritma Support Vector Machine")
        print("C:", self.C)
        print("max_iter:", self.max_iter)
        print("epsilon:", self.epsilon)
        print("kernel: linear")
        print("Index Support Vectors:", self.vektor_pendukung if hasattr(self, 'vektor_pendukung') else "Belum dihitung")
