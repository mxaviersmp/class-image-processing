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
# O arquivo `Trabalho_03.ipynb` possui um notebook executável com os códigos e o relatório do trabalho.
#
# O arquivo `Trabalho_03.pdf` possui o notebook em formato `pdf` o relatório do trabalho.
#
# O arquivo `Trabalho_03.py` possui um script com os códigos do trabalho.
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
# Para ver as opções possíveis para a execução do script, execute `$ python Trabalho_03.py -h`.
#
# Se preferir, abra o notebook e execute uma célula por vez.

import cv2
import requests
import matplotlib.pyplot as plt

import numpy as np

from urllib.parse import urlparse
from pathlib import Path
from PIL import Image
import pytesseract

from skimage.filters import threshold_otsu, sobel
from skimage.transform import rotate, hough_line, hough_line_peaks


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
    plt.show()


# ## Técnica Baseada em Projeção Horizontal

# A detecção de inclinação baseada em projeção horizontal é realizada variando o ângulo testado e proje-
# tando a quantidade de pixels pretos em cada linha linha de texto. O ângulo escolhido é aquele que otimiza
# uma função objetivo calculada sobre o perfil da projeção horizontal. Um exemplo de função objetivo é a
# soma dos quadrados das diferenças dos valores em células adjacentes do perfil de projeção.


def horizontal_projection(img_binary, theta):
    img_rot = rotate(img_binary, theta)
    profile = np.sum(img_rot, axis=1)
    objective_func = np.sum(np.power(np.diff(profile, 1), 2))
    return profile, theta, objective_func


def allign_image_horizontal_projection(image):
    thresh = threshold_otsu(image)
    img_binary = np.where(image > thresh, 0.0, 1.0)
    projections = map(lambda t: horizontal_projection(img_binary, t), range(-90, 90 + 1))
    profile_rot, theta, _ = max(projections, key=lambda x: x[-1])
    img_rot = rotate(image, theta, cval=1)
    img_rot_png = (img_rot * 255).astype(np.uint8)
    return img_rot_png, theta


# **Idéia:**
# Ao testar várias rotações na imagem, calculando a quantidade de píxels contendo informação referentes ao texto em cada linha, identificar o angulo que melhor representa a rotação necesária para alinhar a imagem.

# **Implementação:**
# Para acentuar o texto na imagem, foi calculado um limiar para transformar a imagem em tons de cinza em binária. Esse limiar é calculado utilizando a função `skimage.filters.threshold_otsu`, que utiliza do histograma da imagem para calcular um limiar. Este limiar é então utilizado para zerar os valores abaixo e elevar ao máximo os valores acima, de fato binarizando e invertendo a imagem. Com a imagem ajustada, é calculada a projeção e custo para angulo, da seguinte maneira:
# - Para cada valor de ângulo inteiro $\theta$ de $[-90, 90]$, a imagem é rotacionada em $\theta$ utilizando a função `skimage.transform.rotate`.
# - Com a imagem rotacionada, é calculada a sua projeção na forma da soma de cada linha, aplicando sobre o resultado a função de custo na forma do quadrado da soma das diferenças entre cada linha e seu sucessor.
# - Com o valor de custo calculado para cada ângulo, o $\theta$ com maior valor é escolhido.

# **Chamada e parâmetros:**
# A função `horizontal_projection` recebe a imagem binarizada e o ângulo de rotação, retornando a imagem rotacionada, o perfil de projeção, o ângulo de rotação e o valor.

# A função `allign_image_horizontal_projection` recebe a imagem e retorna esta imagem alinhada e o ângulo utilizado para o alinhamento.

# **Limitações:**
# O ângulo de rotação da imagem deve estar entre $[-90, 90]$ graus.

# **Resultados:**
# Observando as imagens, podemos ver que a técnica foi capaz de alinhar as imagens com rotações positivas e negativas de forma bastante satisfatória. Ao aplicar este alinhamento como uma etapa de pre-processamento para técnicas de `OCR`, observamos que para a imagem 1 original não foi possível obter o seu texto, mas ao aplicar uma rotação de $14°$, foi possível extrair o texto da imagem por completo. Para a imagem 2, é possível perceber alguns erros de reconhecimento, com alguns caracteres faltantes, simbolos e palavras erradas, que são corrigidos ao aplicar uma rotação de $-4°$ a imagem.


# ## Técnica Baseada na Transformada de Hough

# A detecção de inclinação da imagem baseada na transformada de Hough assume que os caracteres de
# texto estão alinhados. As linhas formadas pelas regiões de texto são localizadas por meio da transformada
# de Hough, a qual converte pares de coordenadas (x, y) da imagem em curvas nas coordenadas polares
# (ρ, θ).

# Pixels pretos alinhados na imagem (linhas dominantes) geram picos no plano de Hough e permitem
# identificar o ângulo de inclinação θ do documento. Uma estrutura é empregada para acumular o número
# de vezes em que a combinação de ρ e θ ocorre na imagem. A granularidade do espaço de busca depende
# do grau de precisão do eixo θ.


def hough_transform(img):
    edges = sobel(img)
    hspace, angles_lines, dists_lines = hough_line(edges)
    accum, angles_peaks, dists_peaks = hough_line_peaks(hspace, angles_lines, dists_lines)
    uniques, counts = np.unique(angles_peaks, return_counts=True)
    theta = np.rad2deg(uniques[np.argmax(counts)])
    if theta > 0:
        theta -= 90
    elif theta < 0:
        theta += 90
    return theta, hspace, angles_lines, dists_lines


def allign_image_hough_transform(image):
    theta, hspace, angles_lines, dists_lines = hough_transform(image)
    img_rot = rotate(image, theta, cval=1)
    img_rot_png = (img_rot * 255).astype(np.uint8)
    theta = np.round(theta, 2)
    return img_rot_png, theta


# **Idéia:**
# Ao aplicar a Transformada de Hough em uma imagem com textos, espera-se as linhas horizontais formadas pela transformada seguirão o alinhamento do texto. Utilizando a inclinação destas linhas, podemos encontrar o ângulo de inclinação necessário para alinhar horizontalmente a imagem.

# **Implementação:**
# Assim como na Projeção Horizontal, podemos melhorar o alinhamento acentuando o texto. Para a Transformada de Hough, aplicamos um filtro de detecção de bordas a imagem e fornecemos a saída ao algoritmo. O filtro escolhido foi o Sobel, utilizando a função `skimage.filters.sobel`. Apos o filtro, aplicamos a função `skimage.transform.hough_lines` que aplica a transformada de hough de linhas retas na imagem. A contagem de linhas, seus angulos e distâncias são usados pela função `skimage.transform.hough_line_peaks` para calcular os seus picos no plano de Hough e permitem e os ângulos de inclinação. A partir destes ângulos, o mais frequente é selecionado e transformado de radiano para graus, aplicando um ajuste de $\pm 90°$, pois o ângulo resultante é referente ao eixo vertical.

# **Chamada e parâmetros:**
# A função `hough_transform` recebe a imagem original e retorna o ângulo de rotação, retornando a imagem rotacionada, o acumulador ́da quantiade de vezes que cada pico ocorre na imagem, os ângulos das transformadas e as distancias entre as linhas formadas pelas regiões de texto.

# A função `allign_image_hough_transform` recebe a imagem e retorna esta imagem alinhada e o ângulo utilizado para o alinhamento.

# **Limitações:**
# O ângulo de rotação da imagem deve estar entre $[-90, 90]$ graus.

# **Resultados:**
# Os resultados observados são bastantes similares aos da Projeção Horizontal, com os ângulos encontrados diferindo em casas decimais. Isso ocorre devido a nosso espaço de busca na Projeção Horizontal consistir em apenas inteiros. Na aplicação de `OCR`, pelos angulos serem quase identicos, temos os mesmos resultados, atentanto apenas a imagem 1, que teve uma transcrição referente ao código de barras diferente.


def ocr_images(image, name, image_rot, theta):
    print(f'Texto imagem original {name}: \n{pytesseract.image_to_string(Image.fromarray(image))}')
    print()
    print(f'Texto imagem {name} rotacionada em {theta}°: \n{pytesseract.image_to_string(Image.fromarray(image_rot))}')
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Alinhamento automático de imagens de documentos.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_image', type=str,
        help='url ou caminho para a imagem'
    )
    parser.add_argument(
        'modo',
        choices=['horizontal-projection', 'hough-transform'],
        help='técnica para realizar o alinhamento da imagem'
    )
    parser.add_argument(
        'output_image', type=str,
        help='caminho para salvar a imagem alinhada'
    )
    parser.add_argument(
        '-ocr', action='store_true',
        help='se deve realizar ocr nas imagens'
    )
    args = parser.parse_args()

    image = image_from_url_or_disk(args.input_image)
    if args.modo == 'horizontal-projection':
        img_rot, theta = allign_image_horizontal_projection(image)
    else:
        img_rot, theta = allign_image_hough_transform(image)

    show_images(
        (1, 2),
        [image, img_rot],
        [
            'Original', f'Rotação em {theta}°',
        ],
        'imagem_alinhada'
    )
    if args.ocr:
        ocr_images(image, '', img_rot, theta)

    plt.imsave(Path(args.output_image), img_rot, cmap='gray')
