import gpt4all
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from datasets import load_dataset
from datasets import load_from_disk

class my_model():
    def __init__(self,dataset_dir,index_file):
        self.index_file = index_file
        self.ds = load_from_disk(dataset_dir)
        self.ds.load_faiss_index('embeddings', index_file)
        self.gptj = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy")
        self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    def __call__(self,inp_txt):
        question = inp_txt
        question_embedding = self.q_encoder(**self.q_tokenizer(question, return_tensors="pt"))[0][0].detach().numpy()
        scores, retrieved_examples = self.ds.get_nearest_examples('embeddings', question_embedding, k=5)
        context = "\n".join(retrieved_examples["line"])
        messages = [{"role": "user", "content": f"Please use the following context to answer questions.Context: {context} --- Question: {question}"}]
        ret = self.gptj.chat_completion(messages)
        return ret