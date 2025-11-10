import random
import os
import nltk # Requer: python - import nltk stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
import whisper  # Requer: pip install openai-whisper
from pydub import AudioSegment  # Requer: pip install pydub


# DOWNLOAD DE RECURSOS (SÓ RODA NA 1ª VEZ)

try:
    # Verifica se o 'punkt' (tokenizador) existe
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Baixando 'punkt' do NLTK...")
    nltk.download('punkt')

try:
    # Verifica se as 'stopwords' (palavras de parada) existem
    nltk.data.find('corpus/stopwords')
except LookupError:
    print("Baixando 'stopwords' do NLTK...")
    nltk.download('stopwords')



#  FUNÇÕES DE PROCESSAMENTO


def get_text_from_audio(file_path):
    """
    Recebe um caminho de arquivo (MP3 ou MP4), converte para WAV
    e usa o Whisper para transcrever o texto.
    VERSÃO SIMPLIFICADA SEM MOVIEPY.
    """
    print(f"Processando arquivo: {file_path}")
    
    # Gerar um nome de arquivo WAV temporário
    wav_file = "temp_audio.wav"
    
    # Checar a extensão e converter
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Lógica unificada para MP3 e MP4 usando pydub
    if file_extension == ".mp4" or file_extension == ".mp3":
        print(f"Detectado {file_extension}. Convertendo para WAV usando pydub...")
        try:
            # AudioSegment.from_file lida com múltiplos formatos (incluindo video)
            audio = AudioSegment.from_file(file_path)
            audio.export(wav_file, format="wav")
            
        except Exception as e:
            print(f"Erro ao converter com pydub: {e}")
            print("---")
            print("ERRO COMUM: Você instalou o FFMPEG no seu sistema? (veja choco, brew, apt install ffmpeg)")
            print("---")
            return None
            
    elif file_extension == ".wav":
        print("Detectado WAV. Pulando conversão.")
        wav_file = file_path # Já está no formato certo
    else:
        print(f"Erro: Formato de arquivo '{file_extension}' não suportado.")
        return None

    # Transcrição com Whisper
    print("Iniciando transcrição (isso pode demorar)...")
    model = whisper.load_model("base") 
    result = model.transcribe(wav_file)
    transcribed_text = result["text"]
    
    # Limpar arquivo temporário (se criamos um)
    if file_extension in [".mp3", ".mp4"]:
        os.remove(wav_file)
        
    print("Transcrição concluída.")
    return transcribed_text

def get_summary(text):
    """
    Usa o pipeline 'transformers' para resumir o texto.
    """
    print("Iniciando sumarização do texto...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    print("Sumarização concluída.")
    return summary

def extract_keywords(text, lang='english', min_len=4):
    """
    Usa NLTK para filtrar o texto, removendo stop words e
    palavras curtas, retornando uma lista de palavras-chave.
    """
    print(f"Extraindo palavras-chave (idioma: {lang})...")
    # Define o idioma das stopwords (mude para 'portuguese' se necessário)
    stop_words = set(stopwords.words(lang))
    
    # Separa o texto em palavras (tokens)
    all_words = word_tokenize(text.lower())
    
    keywords = []
    for word in all_words:
        # Verifica se é uma palavra, não é stopword e tem o tamanho mínimo
        if word.isalpha() and word not in stop_words and len(word) >= min_len:
            keywords.append(word.upper())
            
    # Remove palavras duplicadas mantendo a ordem
    unique_keywords = list(dict.fromkeys(keywords))
    print(f"Encontradas {len(unique_keywords)} palavras-chave únicas.")
    return unique_keywords

def create_grid(words, size=15):
    """
    Cria o grid do caça-palavras, agora com verificação
    para não sobrescrever palavras.
    """
    grid = [[" " for _ in range(size)] for _ in range(size)]
    
    for word in words:
        word = word.upper()
        placed = False
        tries = 0
        
        while not placed and tries < 100:
            # 0 = horizontal, 1 = vertical, 2 = diagonal (se quiser adicionar)
            direction = random.choice([(0,1), (1,0)]) # (Vertical, Horizontal)
            
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)

            # Verifica se a palavra cabe
            if (direction == (1,0) and x + len(word) > size) or \
               (direction == (0,1) and y + len(word) > size):
                tries += 1
                continue
                
            # Verifica se o espaço está livre (ou se a letra coincide)
            can_place = True
            for i in range(len(word)):
                char_on_grid = " "
                if direction == (1,0): # Horizontal
                    char_on_grid = grid[y][x+i]
                elif direction == (0,1): # Vertical
                    char_on_grid = grid[y+i][x]
                
                if char_on_grid != " " and char_on_grid != word[i]:
                    can_place = False
                    break
            
            # Se puder posicionar, coloque a palavra no grid
            if can_place:
                for i in range(len(word)):
                    if direction == (1,0): # Horizontal
                        grid[y][x+i] = word[i]
                    elif direction == (0,1): # Vertical
                        grid[y+i][x] = word[i]
                placed = True
            
            tries += 1
            
        if not placed:
            print(f"Aviso: Não foi possível posicionar a palavra '{word}'")

    # Preenche os espaços vazios 
    for i in range(size):
        for j in range(size):
            if grid[i][j] == " ":
                grid[i][j] = chr(random.randint(65, 90)) # Letra maiúscula aleatória
    
    return grid


# EXECUÇÃO PRINCIPAL


def main():
    
    # TRANSCRIÇÃO 
    # Coloque o caminho para o seu arquivo de áudio/vídeo aqui
    # audio_file_path = "s:\Teste.mp3"  # <--- TROQUE AQUI
    # text = get_text_from_audio(audio_file_path)
    # language = 'portuguese' # <--- MUDE SE O ÁUDIO FOR EM PORTUGUÊS

    # Bloco de TEXTO DE EXEMPLO (para testar sem áudio) 
    # Comente ou apague este bloco quando for usar um áudio real
    print("--- USANDO TEXTO DE EXEMPLO ---")
    text = """
    # Quantum computing is an emerging field of technology that leverages the principles of quantum mechanics...
    # (Use o seu texto de exemplo completo aqui se desejar)
    # """
    language = 'english' # O idioma do texto de exemplo
    # Fim do bloco de exemplo 

    if not text:
        print("Nenhum texto foi processado. Encerrando.")
        return

    # RESUMO 
    summary = get_summary(text)
    print("\nResumo Gerado:\n", summary)

    # PALAVRAS-CHAVE 
    keywords = extract_keywords(summary, lang=language, min_len=4)
    if not keywords:
        print("Não foram encontradas palavras-chave. Tente um texto maior ou min_len menor.")
        return
        
    print("\nPalavras-chave filtradas:", keywords)

    # SELEÇÃO DE PALAVRAS 
    try:
        num_words = int(input(f"\nQuantas palavras você quer no caça-palavras? (Max: {len(keywords)}): "))
    except ValueError:
        print("Entrada inválida. Usando 5 palavras.")
        num_words = 5
        
    if num_words > len(keywords):
        print(f"Aviso: Só encontrei {len(keywords)}. Usando todas.")
        random_words = keywords
    else:
        random_words = random.sample(keywords, num_words)

    print("Palavras selecionadas:", random_words)

    # GERAÇÃO DO CAÇA-PALAVRAS
    grid_size = 15 # Ajuste o tamanho do grid se precisar
    print(f"\nGerando grid {grid_size}x{grid_size}...")
    grid = create_grid(random_words, size=grid_size)

    # EXIBIÇÃO 
    print("\n========= CAÇA-PALAVRAS GERADO =========\n")
    for row in grid:
        print(" ".join(row))
    print("\n==========================================")
    print("Palavras para encontrar:", random_words)


# Roda o programa principal
if __name__ == "__main__":
    main()