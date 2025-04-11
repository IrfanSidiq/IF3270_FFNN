# Tugas Besar IF3270 Pembelajaran Mesin - Feed Forward Neural Network

## Deskripsi Singkat
Pada tugas besar ini, peserta kuliah diminta untuk mengimplementasikan FFNN (*Feed-forward Neural Network*) *from scratch*. *Feed-forward Neural Network* adalah jenis arsitektur *neural network* di mana aliran data hanya bergerak satu arah, yaitu dari input layer menuju ke output layer melalui satu atau lebih hidden layer tanpa adanya *loop*. Setiap neuron di dalam layer hanya menerima input dari layer sebelumnya dan mengirimkan output ke layer berikutnya. Proses ini dimulai dengan memasukkan data ke input layer, lalu dikalikan dengan bobot, dijumlahkan, dan diproses melalui fungsi aktivasi di setiap neuron untuk menghasilkan output. FFNN umumnya digunakan dalam tugas-tugas klasifikasi, regresi, dan *pattern recognition*.

## Cara Menjalankan Program
Panduan mengenai contoh cara menjalankan program dapat dilihat di:
- `demo.ipynb` untuk fungsi-fungsi umum dari model FFNN
- `percobaan_laporan.ipynb` untuk percobaan yang dilakukan di laporan

## Pembagian Tugas
| 13522007         | Tensor, Optimizer, WeightInitializer, Layer, FFNN, regularisasi, RMSNorm |
| 13522033         | ActivationFunction (sigmoid, tanh, softmax, GELU), visualisasi model, distribusi bobot dan gradien |
| 13522041         | ActivationFunction (SILU), LossFunction (binary & categorical crossentropy), fitur history & verbose |