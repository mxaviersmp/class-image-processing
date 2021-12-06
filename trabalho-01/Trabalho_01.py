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
# O arquivo `Trabalho_01.ipynb` possui um notebook executável com os códigos e o relatório do trabalho.
#
# O arquivo `Trabalho_01.pdf` possui o notebook em formato `pdf` o relatório do trabalho.
#
# O arquivo `Trabalho_01.py` possui um script com os códigos do trabalho.
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

# # Processamentos Básicos em Imagens Digitais

# ## Transformação de Intensidade
#
# Transformar o espaço de intensidades (nı́veis de cinza) de uma imagem monocromática para:
# - (i) obter o negativo da imagem, ou seja, o nı́vel de cinza 0 será convertido para 255, o nı́vel 1 para
# 254 e assim por diante
# - (ii) converter o intervalo de intensidades para [100, 200].

def negative_image(img):
    return (img * -1) + 255

def clip_image(img, min_v, max_v):
    return np.clip(img, min_v, max_v)


# **Idéia:**
# Para obter o negativo da imagem, basta multiplicar os valores por -1 e somar 255. Isso faz com que os valores próximos a 0 subam para 255 e os mais altos descam a 0, invertendo a intensidade das cores. Para ob
#
# **Implementação:**
# Para criar a imagem negativa, basta multiplicar a imagem por -1 e somar 255.
#
# Para transformar os intervalor, o método `clip` ira cortar os valores fora do intervalo passado.
#
# **Chamada e parâmetros:**
# A função criada recebe a imagem de entrada e retorna o negativo da imagem.
#
# A função criada recebe a imagem e os intervalos mínimo e maximo, retornando a imagem respeitando o novo intervalo
#
#
# **Resultados:**
# Observando o resultado, a imagem negativa apresenta as intensidades invertidas, enquando na imagem transformada é possivel observar um menor contraste e nivel de detalhe devido a faixa de valores possiveis reduzida.

# ## Ajuste de Brilho
#
# Aplicar a correção gama para ajustar o brilho de uma imagem monocromática `A` de entrada e gerar
# uma imagem monocromática `B` de saı́da. A transformação pode ser realizada:
# - (a) convertendo-se as intensidades dos pixels para o intervalo de [0, 255] para [0, 1]
# - (b) aplicando-se a equação $B = A^{(1/γ)}$
# - (c) convertendo-se de volta os valores resultantes para o intervalo [0, 255].

def gamma_correction(img, gamma):
    return (((img / 255) ** (1/gamma)) * 255).astype('uint8')


# **Idéia:**
# Para alterar o fator de brilho da imagem, primeiro é preciso converter a escala para ponto flutuante entre 0 e 1, fazendo isso dividindo pelo valor máximo de intensidade, que é 255 por estar representado em 8 bits.
#
# Com os valores no novo intervalo, após aplicar a formula para ajuste de brilho, as intensidades de cor são convertidas para a representação de 8 bits.
#
# **Implementação:**
# Com o uso dos `arrays` pode realizar cada operação necessária, primeiro normalizando os valores dividindo por 255, após isso aplicando a formula elevando a $(1/Y)$, em seguida retornando a escala original multiplicando por 255, e modificando a representação ao modificar o tipo para inteiro de 8 bits.
#
# **Chamada e parâmetros:**
# A função criada recebe a imagem de entrada e valor de gamma desejado, retornando a imagem transformada com o brilho alterado.
#
# **Resultados:**
# Essa transformação eleva os valores de intensidade de cor dos pixels de acordo com o valor de $γ$, aumentando o valor de intensidade, e com isso a percepção de brilho e claridade na imagem ao aumentar o valor de $γ$.

# ## Quantização de Imagens
#
# Quantização refere-se ao número de nı́veis de cinza usados para representar uma imagem mono-cromática. A quantização está relacionada à profundidade de uma imagem, a qual corresponde
# ao número de bits necessários para armazenar a imagem. Represente uma imagem com diferentes
# nı́veis de quantização.

def quantitize_image(img, levels):
    if not (0 < levels <= 256):
        raise ValueError('`levels` deve estar ]0, 256]')
    quantitize = np.arange(0, 256, int(256 / levels))
    quantitized = quantitize[np.digitize(img, quantitize) - 1]
    quantitized = (quantitized / quantitize.max()) * 255
    return quantitized.astype('uint8')


# **Idéia:**
# Quantizar a imagem é aplicar uma transformação da escala de cinza, no caso de imagens monocromáticas. Essa transformação altera a percepção da quantidade de bits utilizada para representar as cores em uma imagem.
#
# Para alcançar este efeito, podemos criar uma distribuição dos valores de intensidade da imagem original, incrementando os passos de acordo com o nível desejado. Isso significa que, como uma imagem de 8bits pode assumir 256 níveis, criamos uma distribuição de 0 a 255. Reduzir para uma representção de 4bits significa que cada píxel pode possuir 16 níveis de intensidade. Com isso, modificando o passo da distribuição alteramos os valores que cada pixel pode assumir. Por exemplo, se 8 bits pode assumir valores `[0, 1, 2, ..., 254, 255]`, 4 bits pode assumir os valores `[0, 16, 32, ..., 240, 255]`. Com essas novas faixas de valores, podemos transformar cada píxel da imagem original no valor correspondente na nova representação. Assim, se uma faixa da imagem de 8 bits possui valores `[76, 114,  46,  46,  97]`, sua representaão em 4 bits seria `[ 64, 112,  32,  32,  96]`.
#
# **Implementação:**
# Para a implementação utilizando o `numpy`, é criada um arranjo de `0 a 256`, com um passo de acordo com o nível desejado. Esse arranjo é utilizado para mapear os valores da imagem original as novas intensidades, com a ajuda do método `digitize`, que retorna os índices das faixas aos quais cada valor na matriz de entrada pertence. Com os valores quantizados, é aĺicada uma normalização para transformar os valores na faixa de 0 a 255.
#
# **Chamada e parâmetros:**
# A função criada recebe a imagem de entrada e o nível de intensidade desejado, retornando a imagem transformada com
# os novos valores.
#
# **Limitações:** O valor de nivel de respeitar o intervalo $]0, 256]$.
#
# **Resultados:**
# Como observado no resultado, reduzir a quantidade de bits resulta em uma menor variabilidade de cores, criando uma imagem menos detalhada.

# ## Planos de Bits
#
# Extrair os planos de bits de uma imagem monocromática. Os nı́veis de cinza de uma imagem
# monocromática com m bits podem ser representados na forma de um polinômio de base 2:
#
# $a_{m−1} 2^{m−1} + a_{m−2} 2^{m−2} + . . . + a_1 2^{1} + a_0 2^{0}$
#
# O plano de bits de ordem 0 é formado pelos coeficientes a 0 de cada pixel, enquanto o plano de
# bits de ordem $m − 1$ é formado pelos coeficientes $a_{m−1}$.

def bitplan_image(img, bp):
    h, w = img.shape
    img = np.unpackbits(img.reshape(-1, 1), axis=1, bitorder='little')
    if not (0 <= bp <= 7):
        raise ValueError('bp deve estar entre [0, 7]')
    bitplan = img[:, bp] * (2**bp)
    bitplan = ((bitplan / 2**bp) * 255).astype('uint8')
    return bitplan.reshape(h, w)


# **Idéia:**
# O plano de bits é um corte na imagem, extraindo o valor representado pelo bit correspondente. Isso significa que, se o pixel de uma imagem possui o valor $137$, em 8 bits, sua representação binária é $10001001$. Ao cortar essa imagem no plano 3, o valor desse pixel será representado pelo valor do bit 3, multiplocado pela a potencia $2^3$, que no caso seria 8. Assim, o valor máximo de intensidade dessa imágem é 8.
#
# **Implementação:**
# Utilizamos do método `unpackbits` para transformar cada valor de pixel em nossa imagem em um array com sua representação binária. Para isso, modificamos o formato de nossa imagem, isolando cada pixel em uma linha. Para facilitar a aritmética, escolhemos representar de forma binária little-endian. Com a imagem representada em bits, escolhemos o bit do plano desejado, multiplicando pela potencia de 2. O resultado é então normalizado e reformatado para as dimensões da imagem.
#
# **Chamada e parâmetros:**
# A função criada recebe a imagem de entrada e o plano que deseja realizar o corte, retornando a imagem o corte nesse plano.
#
# **Limitações:** O valor do plano de bits de respeitar o intervalo $[0, 7]$.
#
# **Resultados:**
# Como observado no resultado, cada plano exibe apenas pixels que possuem o bit representado naquele nivel. No caso do plano 0, pixel pares são exibidos.

# ## Mosaico
#
# Construir um mosaico de 4 × 4 blocos a partir de uma imagem monocromática. A disposição dos
# blocos deve seguir a numeração mostrada:
#
# |    |    |    |    |   |   |   |    |    |    |   |
# |----|----|----|----|---|---|---|----|----|----|---|
# | 1  |  2 |  3 |  4 |   |   |   | 6  | 11 | 13 | 3 |
# | 5  |  6 |  7 |  8 |   | &rarr;  |   | 8  | 16 | 1  | 9 |
# | 9  | 10 | 11 | 12 |   |   |   | 12 | 14 | 2  | 7 |
# | 13 | 14 | 15 | 16 |   |   |   | 4  | 15 | 10 | 5 |

def mosaic_image(img, n_blocks, new_order):
    h, w = img.shape

    if h != w:
        raise ValueError('Imagem deve ser quadrada.')
    if n_blocks <= 0 or (h * w) % n_blocks > 0:
        raise ValueError('numero de blocos dever ser divisor das dimensões')

    block_size = h // n_blocks

    shape = (
        h // block_size,
        w // block_size,
        block_size,
        block_size,
    )
    strides = (
        w * block_size,
        block_size,
        w,
        1
    )

    blocks = np.lib.stride_tricks.as_strided(
        img,
        shape=shape,
        strides=strides
    )
    blocks = blocks.reshape(-1, block_size, block_size)
    blocks = blocks[new_order]
    blocks = blocks.reshape(n_blocks, n_blocks, block_size, block_size)
    blocks = blocks.swapaxes(1, 2)

    return blocks.reshape(h, w)


# **Idéia:**
# Dividir a imagem em diferentes segmentos quadriculares, reorganizando esses segmentos.
#
# **Implementação:**
# Primeiramente é preciso segmentar a imagem em blocos. Para isso, podemos dividir a imagem em novas regiões, para que possamos acessar cada região como uma matrix. Assim, cada indice da nova matriz de regioes contem um segmento da imagem. Esta segmentação não é trivial pois a imagem é representaga em segmentos contínuos de pixels. Para realizar essa divisão em blocos, precisamos modificar como os arrays. O `numpy` utiliza de passos, ou `strides`, para determinar onde termina cada linha. Modificando esse valor, em conjunto com o formato, é possivel realizar essa divisão em blocos.
#
# O novo tamanho da imagem será o tamanho da matriz de blocos e o tamanho dos blocos, `(# blocos, # blocos, tamanho blocos, tamanho blocos)`. Uma imagem 8 * 8, dividida em 4 blocos, resultaria em um segmentação tamanho `(4, 4, 2, 2)`, acessando o segmento `[0, 0]` retornaria o bloco superior direito, de tamanho `(2, 2)`. Para os passos, o passo inicial é a largura pelo tamanho do bloco, o segundo passo é o tamanho de cada bloco, o terceiro é a largura da imagem, com o quanto sendo o canal de cor. No exemplo, o primeiro passo tem tamanho 16, que é a `distancia de um bloco a seu adjacente vertical`. O segundo passo tem tamanho 2, a `distancia de um bloco a seu adjacente horizontal`. O terceiro é a `distancia de um pixel ao proximo em cada linha`.
#
# Com a imagem segmentada, transformamos a matrix em um vetor, apenas para facilitar sua reordenação. Essa ordem é dada por uma sequencia de numeros determinando o indice de cada segmento. Com o mosaico reorganizado, agora é feita a união da imagem. Primeiro o formato é transfomado novamente em uma matriz de segmentos.
# Para 'colar' a imagem, é preciso inverter a ordem dos eixos dos blocos, trocando o segundo com o terceiro eixo. Por fim, modificamos novamente o formato da imagem para o original.
#
# **Chamada e parâmetros:**
# A função criada recebe a imagem de entrada, a quantidade de blocos por linha e a nova ordenação do mosaco, retornando o novo mosaico.
#
# **Limitações:** A imagem deve ser quadrada e o numero de blocks deve ser um divisor das dimensões.
#
# **Resultados:** Como observado no resultado, a imagem foi segmentada em blocos, reordenados de acordo com a escolha do usuário.

# ## Combinação de Imagens
#
# Combinar duas imagens monocromáticas de mesmo tamanho por meio da média ponderada de
# seus nı́veis de cinza

def combine_images(img1, f1, img2, f2):
    if img1.shape != img2.shape:
        raise ValueError('Imagens devem possuir mesmas dimensões.')
    return (np.clip(img1 * f1, 0, 255) + np.clip(img2 * f2, 0, 255)).astype('uint8')


# **Idéia:**
# Combinar as imagens de modo que a imagem resultado possua uma porcentagem de cada.
#
# **Implementação:**
# Primeiramente é multiplicada cada imagem por seu fator, limitando o resultado a faixa de valores de intensidade. Apos isso, as imagens são combinadas apenas somando ambas.
#
# **Chamada e parâmetros:**
# As imagens e seus fatores de multiplicação, com imagem 1 e fator 1, imagem 2 e fator 2.
#
# **Limitações:** Imagens devem possuir as mesmas dimensões.
#
# **Resultados:** Como observado no resultado, as imagems resultantes são resultado de uma combinação entre as duas. A imagem com maior peso é mais prevalente e visível no resultado.

# ## Filtragem de Imagens
#
# Uma operação de filtragem aplicada a uma imagem digital é altera localmente os valores de intensidade dos pixels da imagem levando-se em conta tanto o valor do pixel em questão quanto
# valores de pixels vizinhos. No processo de filtragem, utiliza-se uma operação de convolução de
# uma máscara pela imagem. Este processo equivale a percorrer toda a imagem alterando seus
# valores conforme os pesos da máscara e as intensidades da imagem.
#
# Aplicar os seguintes filtros (individualmente) em uma imagem digital monocromática:
#
# |    |    |    |    |   |   |    |    |    |    |
# |----|----|----|----|---|---|----|----|----|----|
# |    | -1 | -1 | -1 |   |   |    | -1 | -2 | -1 |
# | h1 | -1 | 8  | -1 |   |   | h2 | 0  | 0  | 0  |
# |    | -1 | -1 | -1 |   |   |    | 1  | 2  | 1  |

def convolution(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape

    if kh != kw:
        raise ValueError('kernel deve ser quadrado')

    kernel = np.flipud(np.fliplr(kernel))
    convoluted_img = np.zeros_like(img)

    k = (kh - 1) // 2

    image_padded = np.zeros((h + kh - 1, w + kw - 1))
    image_padded[k:-k, k:-k] = img

    for x in range(w):
        for y in range(h):
            convoluted_img[y, x] = (kernel * image_padded[y: y + kh, x: x + kw]).sum()

    return convoluted_img


# **Idéia:**
# Aplicar filtros as imagens com o uso de convoluções.
#
# **Implementação:**
# Primeiramente, para calcular a convolução, o filtro é espelhado no centro. Após isso é adicionado um padding a imagem, do tamanho do filtro menos uma unidade. Então são realizadas as convoluções. Para cada indice na imagem, são escolhidos os indices correspondentes na região de vizinhança do tamanho do filtro. Esta região é multiplicada pelo filtro, o resultado é somado e atribuido ao pixel correspondente da imagem com a convolução.
#
# **Chamada e parâmetros:**
# A função criada recebe a imagem de entrada e o filtro que deseja ser aplicado, retornando a nova imagem 'filtrada'.
#
# **Limitações:** O filtro deve ser quadrado.
#
# **Resultados:**
# O resultado mostra a imagem original e os resultados dos filtros. Apesar do grande ruído, o filtro `h1` aparenta detectar as bordas das imagem, enquanto o filtro `h2` aparenta detectar arestas horizontais.

# ## Entropia
#
# Calcular a entropia de uma imagem monocromatica, de acordo com a equação:
#
# $H = −\sum^{Lmax}_{i=0}p_i \log p_i$
#
# em que a distribuição dos nı́veis de intensidade da imagem pode ser transformada em uma função
# densidade de probabilidade, dividindo-se o número de pixels de intensidade $i$, denotado $n_i$,
# pelo número total $n$ de pixels na imagem, ou seja $p_i = \dfrac{n_i}{n}$, em que $\sum^{Lmax}_{i=0}p_i = 1$

def image_entropy(img):
    if len(img.shape) > 2:
        raise ValueError('Imagem deve possuir apenas 1 canal de cor')
    hist, _ = np.histogram(img, bins = 256)
    hist = hist / img.size
    hist_non_zero = hist[hist > 0]
    return -np.sum(np.multiply(hist_non_zero, np.log2(hist_non_zero)))

# **Idéia:**
# Calcular a entropia das imagens em escala de cinza.
#
# **Implementação:**
# Primeiramente, é calculado o histograma da imagem, para determinar o quanto de cada intensidade de cinza está presente na imagem. Apos isso, os valores do histograma são divididos pelo tamanho da imagem. As intensidades com valor 0 são removidas do histograma. Por fim, a formula para o calculo da entropia é utilizada.
#
# **Chamada e parâmetros:**
# A imagem para calcular, retornando sua entropia.
#
# **Limitações:** A imagem deve possuir apenas 1 canal de cor.
#
# **Resultados:**
# Pelo resultado, podemos inferir que a imagem `baboon` possui mais informação, ou mais detalhes, que a `butterfly`.


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Processamentos Básicos em Imagens Digitais.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'imagens', type=str, nargs='*',
        help='url ou caminho para as duas imagensimagem'
    )
    parser.add_argument(
        '--n_blocos', type=int, nargs='?', default=4,
        help='numero de blocos para mosaico na questão 5'
    )
    parser.add_argument(
        '--ordem_mosaico', type=list, nargs='?',
        default=[
            [6, 11, 13, 3],
            [8, 16, 1, 9],
            [12, 14, 2, 7],
            [4, 15, 10, 5]
        ],
        help='ordenação do mosaico na questão 5'
    )
    parser.add_argument(
        '--h1', type=list, nargs='?',
        default=[
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ],
        help='filtro h1 na questão 7'
    )
    parser.add_argument(
        '--h2', type=list, nargs='?',
        default=[
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],

        ],
        help='filtro h2 na questão 7'
    )
    args = parser.parse_args()
    if args.imagens and len(args.imagens) not in (0, 2):
        parser.error('Passe duas imagens, ou não passe nenhuma para usar as padrões')

    # img1 = 'https://www.ic.unicamp.br/~helio/imagens_png/baboon.png'
    # img2 = 'https://www.ic.unicamp.br/~helio/imagens_png/butterfly.png'
    img1 = './bike.jpeg'
    img2 = './engine.jpeg'

    if args.imagens:
        img1 = args.imagens[0]
        img2 = args.imagens[1]

    image_1 = image_from_url_or_disk(img1)
    image_2 = image_from_url_or_disk(img2)
    empty_img = np.full_like(image_1, 255, dtype=np.uint8)

    show_images((1, 2), [image_1, image_2], ['imagem 1', 'imagem 2'], 'imagens_originais')

    show_images(
        (1, 3),
        [image_1, negative_image(image_1), clip_image(image_1, 100, 200)],
        ['original', 'negativo da imagem', 'imagem transformada'],
        'q1_transformacao_de_intensidade'
    )

    show_images(
        (1, 4),
        [image_1] + [gamma_correction(image_1, gamma) for gamma in [1.5, 2.5, 3.5]],
        ['original', 'γ = 1.5', 'γ = 2.5', 'γ = 3.5'],
        'q2_ajuste_de_brilho'
    )

    show_images(
        (3, 3),
        [empty_img, image_1, empty_img] + [quantitize_image(image_1, level) for level in [64, 32, 16, 8, 4, 2]],
        ['', '256 níveis', '', '64 nı́veis', '32 nı́veis', '16 nı́veis', '8 nı́veis', '4 nı́veis', '2 nı́veis'],
        'q3_quantizacao_de_imagens'
    )

    show_images(
        (1, 4),
        [image_1] + [bitplan_image(image_1, bp) for bp in [0, 4, 7]],
        ['original','plano de bit 0', 'plano de bit 4', 'plano de bit 7'],
        'q4_planos_de_bits'
    )

    n_blocks = args.n_blocos
    new_order = np.array(args.ordem_mosaico) - 1
    image_1_mosaic = mosaic_image(image_1, n_blocks, new_order)
    show_images(
        (1, 2),
        [image_1, image_1_mosaic],
        ['original', 'mosaico'],
        'q5_mosaico'
    )

    show_images(
        (2, 3),
        [image_1, empty_img, image_2] + [
            combine_images(image_1, f1, image_2, f2)
            for f1, f2 in [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]
        ],
        ['A', '', 'B', '0.2*A + 0.8*B', '0.5*A + 0.5*B', '0.8*A + 0.2*B'],
        'q6_combinacao_de_imagens'
    )

    h1 = np.array(args.h1)
    h2 = np.array(args.h2)
    show_images(
        (2, 3),
        [
            image_1, convolution(image_1, h1), convolution(image_1, h2),
            image_2, convolution(image_2, h1), convolution(image_2, h2),
        ],
        [
            'imagem 1', 'imagem 1 h1', 'imagem 1 h2',
            'imagem 2', 'imagem 2 h1', 'imagem 2 h2'
        ],
        'q7_filtragem_de_imagens'

    )
    print(f'A entropia da imagen imagem 1 é {image_entropy(image_1)}')
    print(f'A entropia da imagen imagem 2.png é {image_entropy(image_2)}')
