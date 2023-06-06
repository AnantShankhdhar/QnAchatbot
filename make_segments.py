import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import argparse

parser = argparse.ArgumentParser(description='Breaking Documents into Segments')

model = SentenceTransformer('all-mpnet-base-v2')

def get_seg_list(page):
    text = page
    sentences = text.split('. ')
    # Embed sentences
    embeddings = model.encode(sentences)
    # Create similarities matrix
    similarities = cosine_similarity(embeddings)
    activated_similarities = activate_similarities(similarities, p_size=embeddings.shape[0])
    minmimas = argrelextrema(activated_similarities, np.less, order=2)
    sentece_length = [len(each) for each in sentences]
    # Determine longest outlier
    long = np.mean(sentece_length) + np.std(sentece_length) *2
    # Determine shortest outlier
    short = np.mean(sentece_length) - np.std(sentece_length) *2
    # Shorten long sentences
    text = ''
    for each in sentences:
        if len(each) > long:
            # let's replace all the commas with dots
            comma_splitted = each.replace(',', '.')
        else:
            text+= f'{each}. '
    sentences = text.split('. ')
    # Now let's concatenate short ones
    text = ''
    for each in sentences:
        if len(each) < short:
            text+= f'{each} '
        else:
            text+= f'{each}. '
    split_points = [each for each in minmimas[0]]
    # Create empty string
    text = ''
    seg_list = list()
    for num,each in enumerate(sentences):
        # Check if sentence is a minima (splitting point)
        if num in split_points:
            # If it is than add a dot to the end of the sentence and a paragraph before it.
            seg_list.append(text)
            text+=""
        else:
            # If it is a normal sentence just add a dot to the end and keep adding sentences.
            text+=f'{each}. '
    return seg_list

def rev_sigmoid(x:float)->float:
    return (1 / (1 + math.exp(0.5*x)))
    
def activate_similarities(similarities:np.array, p_size=1)->np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum 
        Returns:
            list: list of weighted sums
        """
        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10,10,p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid) 
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1,1)
        ### 5. Calculate the weighted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities

def get_avg_tokens(txt_list):
    word_count = []
    for i in txt_list:
        txt = i.split()
        word_count.append(len(txt))
    return np.median(word_count)

dir_name = "data/"
text_list = list()
for file in os.listdir(dir_name):
    text_file = list()
    reader = PdfReader(dir_name+file)

    # getting a specific page from the pdf file
    for i in range(len(reader.pages)):
        page = reader.pages[i]

        # extracting text from page
        text = page.extract_text()
        text_file.append(text)
    text_list.append(text_file)

torch.set_grad_enabled(False)
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


segments = []
for doc in text_list:
    for page in doc:
        segments.extend(get_seg_list(page))

sentence_list = []
segment_list = []
for string in segments:
    encoding = ctx_tokenizer.encode(string)
    if len(encoding) > 500:
        first_part = encoding[:500]
        second_part = encoding[500:]
        sentence_list.append(ctx_tokenizer.decode(first_part))
        sentence_list.append(ctx_tokenizer.decode(second_part))
        segment_list.append(string)
        segment_list.append(string)
    else:
        sentence_list.append(string)
        segment_list.append(string)

data_dict = {}
data_dict['line'] = sentence_list
data_dict['segment'] = segment_list

ds = Dataset.from_dict(data_dict)
df.to_csv('data/segments.csv', index=False)
ds_with_embeddings = ds.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example["line"], truncation = True,return_tensors="pt"))[0][0].numpy()})
ds_with_embeddings.add_faiss_index(column='embeddings')
ds_with_embeddings.save_to_disk('data/ds_with_embeddings')
ds_with_embeddings.save_faiss_index('embeddings', 'data/my_index.faiss')
