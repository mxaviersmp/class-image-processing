#!/usr/bin/env python
# coding: utf-8

# MC920 / MO443 (Introdução ao Processamento Digital de Imagem)
#
# Prof. Hélio Pedrini
#
# Matheus Xavier Sampaio - RA 220092

# # Instruções
#
# ### Arquivos
#
# O arquivo `Trabalho_02.ipynb` possui um notebook executável com os códigos e o relatório do trabalho.
#
# O arquivo `Trabalho_02.pdf` possui o notebook em formato `pdf` o relatório do trabalho.
#
# O arquivo `Trabalho_02.py` possui um script com os códigos do trabalho.
#
# ### Ambiente e Execução
#
# O arquivo `environment.yml` pode ser utilizado para criar um ambiente `conda` com todas as dependencias para executar o trabalho.
#
# Caso tenha o Anaconda instalado, basta executar o comando: `$ conda env create`
#
# Apos instalar o ambiente, ative usando o comando: `$ conda activate mo443`
#
# Por fim, execute o script python para criar as imagens com a visualização de cada questão: `$ python Trabalho_01.py`.
# Para ver as opções possíveis para a execução do script, execute `$ python Trabalho_01.py -h`.
#
# Se preferir, abra o notebook e execute uma célula por vez.

import cv2
import requests
import matplotlib.pyplot as plt
import numpy as np

from urllib.parse import urlparse
from pathlib import Path


plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def image_from_url_or_disk(path):
    if urlparse(path).scheme:
        print(f'baixando imagem de: {path}')
        resp = requests.get(path)
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    elif Path(path).exists():
        print(f'abrindo imagem de: {path}')
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError('caminho deve ser url ou arquivo')
    return image


def show_images(grid, images, titles, filepath):
    plt.figure(figsize=(19, 10))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(*grid, i)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')
    plt.suptitle(filepath)
    plt.savefig(f'{filepath}.png', transparent=False)
    plt.show()

# # # Transformada rápida de Fourier

# ## Espectro de Fourier
#
# Aplicar a transformada rápida de Fourier (do inglês, Fast Fourier Transform - FFT)
# em imagens digitais, convertendo-as para o domı́nio de frequência.

def image_fft_spectrum(image):
    img_fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    img_fft_shift = np.fft.fftshift(img_fft)
    magnitude_spectrum = cv2.magnitude(img_fft_shift[:,:,0], img_fft_shift[:,:,1])
    magnitude_spectrum = 20 * np.log(magnitude_spectrum)
    return img_fft_shift, magnitude_spectrum

def fft_image(img_fft_shift, normalize=True):
    f_ishift = np.fft.ifftshift(img_fft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    if normalize:
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

# **Idéia:**
# É possível utilizar das implementações da Transformada discreta de Fourier para calcular o espectro de fourrier de uma imagem.
# Com isso, é possível visualizar a imagem no domínio de frequência, e visualizar a imagem no domínio do espaço após aplicar algum filtro no domínio da frequência.

# **Implementação:**
# Primeiramente é aplicada a Transformada discreta de Fourier na imagem (`cv2.dft`), então deslocando o componente de frequência para o centro do espectro (`np.fft.fftshift`).
# Apos isso, é calculada a magnitude de cada ponto (`cv2.magnitude`), aplicando o log as magnitudes para reduzir a distorção e amplificar a diferença entre as magnitudes (`np.log`).

# Com a transformada e o espectro calculados, para converter a transformada para imagem no domínio de espaçço, é aplicada o inverso da centralização do componente (`np.fft.ifftshift`), o inverso da Transformada discreta de Fourier (`cv2.idft`), calculando as magnitudes e normalizando a imagem.

# **Chamada e parâmetros:**
# A função `image_fft_spectrum` recebe a imagem original, retornando a imagem transformada e o espectro de magnitude.

# A função `fft_image` recebe uma imagem transformada e retorna a imagem correspondente a transformação.

# **Resultados:**
# A imagem do espectro mostra uma visualização da imagem no domínio da frequência. É possível observar maior intensidade no centro do espectro, o que indica maior concentração de frequencias mais baixar, mas alguns pontos nas frequências mais altas podem ser observados. Outra observação são as linhas horizontal e vertical cortando o centro do espectro, indicando bordas horizontais e verticais no domínio do espaço, com mais prevalencia de bordas horizontais.
# Converter o domínio de frequência para o domínio do espaço sem aplicar nenhum filtro resulta na imagem original.

# ## Núcleo dos filtros
#
# Filtros passa-baixa, passa-alta, passa-faixa e rejeita-faixa em imagens monocromáticas

def kernels(image, radius, thickness):
    w, h = image.shape

    passa_baixa = cv2.circle(np.zeros_like(image), (h//2, w//2), radius, 1, -1).astype(np.uint8)
    passa_alta = (passa_baixa * -1) + 1

    passa_faixa = cv2.circle(np.zeros_like(image), (h//2, w//2), radius, 1, thickness).astype(np.uint8)
    rejeita_faixa = (passa_faixa * -1) + 1
    
    return passa_baixa, passa_alta, passa_faixa, rejeita_faixa

# **Idéia:**
# Para criar os filtros é possível utilizar implementações de criar círculos. Sobrepondo esses filtros no espectro de magnitude permite visualizar as frequências que serão filtradas.

# **Implementação:**
# Calculando o centro da imagem, é utilizada a função `cv2.circle` para criar os núcleos dos filtros. Para o passa baixa, é criado um círculo cheio, já para o passa baixa é criado um círculo com bordas, que permite excluir a faixa central. Os filtros passa alta e rejeita faixa são apenas o inverso de seus correspondentes.

# **Chamada e parâmetros:**
# A função recebe a visualização do espectro de magnitude, o raio e a espessura do circulo, que serão utilizados para criar os núcleos dos filtros.

# **Resultados:**
# Observando as imagens é possível visualizar os filtros, e quais frequencias serão aceitas.

# ## Filtragem das imagens no domı́nio de frequência
#
# Possibilita a alteração de seus valores originais em novas
# informações, de forma a atenuar ruı́do nas imagens, suavizar os dados, aumentar o contraste,
# realçar detalhes (bordas) das imagens, entre outras operações

def filtragem(img_fft_shift, kernel):
    if img_fft_shift.shape[:2] != kernel.shape:
        raise ValueError('Transformada e filtro devem possuir dimensões compatíveis.')
    mask = np.zeros_like(img_fft_shift, np.uint8)
    mask[:,:,0] = kernel
    mask[:,:,1] = kernel
    return fft_image(img_fft_shift * mask)

# **Idéia:**
# Usando os filtros, é possível modificar o domínio da frequência, aplicando as transformações e modificando a imagem no domínio do espaço.

# **Implementação:**
# É criada uma mascara nas mesmas dimensões da transformada, e copiar o filtro para cada dimensão.
# Com isso, é multiplicada a transformada pela mascara e convertido o resultado para o domínio do espaço.

# **Chamada e parâmetros:**
# A função recebe a imagem transformada e o filtro que será utilizado.

# **Limitações:**
# O filtro deve possuir dimensão compatível com a transformada. Como a transformada possui dimensões `(h, w, 2)`, o filtro deve possuir dimensão `(h, w)`.

# **Resultados:**
# Na imagem filtrada com passa baixa, é possível observar um 'borramento' da imagem pelo enfraquecimento das bordas, o que se dá pelo filtro remover frequencias mais altas, geralmente encontradas nas bordas da imagem.
# Na imagem filtrada com passa alta, visualizamos as bordas e transições que foram excluídas com o filtro passa baixa, observadas na transição dos olhos e nariz, assum como na pelugem.
# Na imagem filtrada com passa faixa, devido a configuração do filtro, observamos a formação da imagem com algumas sem as frequencias mais baixas, e a perda de algumas transições devido a remoção das frequências mais altas.
# Na imagem filtrada com rejeita faixa, devido a configuração do filtro, com a remoção das freqências 'intermediárias', observamos a imagem borrada, dexido as frequencias baixas, com uma sobreposição das bordas e transições de forma atênuada, devido as frequências mais altas.

# ## Compressão de imagem
#
# Diferentes estratégias podem ser aplicadas às imagens, tal
# como a remoção de coeficientes cujas magnitudes são menores do que um determinado limiar
# (atribuindo-se valores iguais a 0 a eles). Apresente os histogramas das imagens antes e após
# a compressão.

def compress_image_fft(img_fft_shift, tresh):
    if not 0 <= tresh <= 100:
        raise ValueError('Limiar deve estar entre 0 e 100')
    abs_frequencies = np.abs(img_fft_shift)
    tresh_mask = abs_frequencies >= np.percentile(abs_frequencies.flatten(), 100 - tresh)
    compressed_img = fft_image(img_fft_shift * tresh_mask)
    return tresh_mask, compressed_img

# **Idéia:**
# É possível comprimir a imagem manendo as `n` porcento magnitudes com menores coeficientes, permitindo observar o quanto de informação da imagem pode ser mantido.

# **Implementação:**
# A partir da transformada é calculado o valor absoluto de cada coeficiente e ordenados apos redimensionar para 1 dimensão.
# Com isso, encontramos o valor que representa o `n` percentil, sendo este o limiar. Esse valor é utilizado para criar uma mascara que irá filtrar todas as frequências com valor inferiores a este.
# A mascara é então usada para filtrar o domínio da frequencia, que será convertido para o domínio do espaço, resultando na imagem com menos informação.

# **Chamada e parâmetros:**
# Transformada da imagem e limiar de compressão.

# **Limitações:**
# Limiar deve estar entre 0 e 100.

# **Resultados:**
# A imagem comprimida demonstra que, mantendo apenas 0.5% da informação contida na imagem original ainda é possível distiguir bem a imagem, enquanto os histogramas exibem como a distribuição de pixels muda da imagem original para a comprimida.


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Transformada rápida de Fourier.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--imagem', type=str, nargs='?', default='./bike.jpeg',
        help='url ou caminho para a imagem'
    )
    parser.add_argument(
        '--circle_radius', type=int, nargs='?', default=60,
        help='tamanho do circulo para nucleos dos filtros'
    )
    parser.add_argument(
        '--circle_thickness', type=int, nargs='?', default=80,
        help='tamanho do centro circulo para nucleos dos filtros faixa'
    )
    parser.add_argument(
        '--treshold', type=float, nargs='?', default=0.5,
        help='limiar para compressão de imagem, entre 0 e 100'
    )
    args = parser.parse_args()

    if not 0 <= args.treshold <= 100:
        parser.error('Limiar deve estar entre 0 e 100')

    img = args.imagem
    radius = args.circle_radius
    thickness = args.circle_thickness
    tresh = args.treshold

    image = image_from_url_or_disk(img)
    empty_img = np.full_like(image, 255, dtype=np.uint8)

    show_images((1, 1), [image], [''], 'imagem_original')

    img_fft_shift, magnitude_spectrum = image_fft_spectrum(image)
    img_back_norm = fft_image(img_fft_shift)
    show_images(
        (1, 3), 
        [image, magnitude_spectrum, img_back_norm],
        ['imagem original', 'espectro de Fourier', 'imagem após'],
        'imagem_original_e_transformada'
    )
    
    passa_baixa, passa_alta, passa_faixa, rejeita_faixa = kernels(image, radius, thickness)
    show_images(
        (1, 4), 
        [
            magnitude_spectrum * passa_baixa, magnitude_spectrum * passa_alta,
            magnitude_spectrum * passa_faixa, magnitude_spectrum * rejeita_faixa
        ],
        ['núcleo passa baixa', 'núcleo passa alta', 'núcleo passa faixa', 'núcleo rejeita faixa'],
        'nucleos_filtros'
    )
    
    show_images(
        (1, 4), 
        [
            filtragem(img_fft_shift, passa_baixa),
            filtragem(img_fft_shift, passa_alta),
            filtragem(img_fft_shift, passa_faixa),
            filtragem(img_fft_shift, rejeita_faixa)
        ],
        [
            'imagem após filtragem passa baixa',
            'imagem após filtragem passa alta',
            'imagem após filtragem passa faixa',
            'imagem após filtragem rejeita faixa'
        ],
        'filtragens'
    )
    
    tresh_mask, compressed_img = compress_image_fft(img_fft_shift, tresh)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(magnitude_spectrum, cmap='gray', vmin=0, vmax=255)
    plt.title('espectro de fourier')
    plt.subplot(1, 2, 2)
    plt.imshow(np.any(tresh_mask, axis=-1) * magnitude_spectrum, cmap='gray')
    plt.title('mascara de limiar para compressão')

    plt.show()

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title('imagem original')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.hist(image.flatten(), bins=256);
    plt.title('histograma da imagem original')

    plt.subplot(2, 2, 3)
    plt.imshow(compressed_img, cmap='gray', vmin=0, vmax=255)
    plt.title(f'imagem comprimida mantendo {tresh}%')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.hist(compressed_img.flatten(), bins=256)
    plt.title('histograma da imagem comprimida')

    plt.suptitle('compressao_de_imagem')
    plt.savefig('compressao_de_imagem.png', transparent=False)
    plt.show()
