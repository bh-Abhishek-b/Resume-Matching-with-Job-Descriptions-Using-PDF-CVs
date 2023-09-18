import pandas as pd
import numpy as np
from transformers import DistilBertModel,DistilBertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-uncased')                                      # Creating a tokenizer from a pretrained model (distilbert-base-uncased)
model=DistilBertModel.from_pretrained('distilbert-base-uncased')                                              # Selecting the model
job_description_embeddings=[]
cv_embeddings=[]
with open('/home/mav_27/extracted_data.txt','r') as file:                                                     # Opening the extracted data file created from extractor
    for line in file:
        cv_details=tokenizer(line,return_tensors='pt',padding=True,truncation=True,max_length=512)            # Tokenizing cv_details
        with torch.no_grad():
          output = model(**cv_details)
    embeddings = output.last_hidden_state.mean(dim=1).numpy()                                                 # Calculating the mean of the last hidden state across the first dimension (dim=1)
    cv_embeddings.append(embeddings)                                                                          # Appending the mean embedding to the list


df=pd.read_csv('/home/mav_27/Internship Assignment/sample_df.csv')                                            # Opening the data frame created for Job Description extraction 
for line in df.job_description:
    job_descript=tokenizer(line,return_tensors='pt',padding=True,truncation=True,max_length=512)              # Tokenizing the Job Descriptions
    with torch.no_grad():
        output = model(**job_descript)
    embeddings = output.last_hidden_state.mean(dim=1).numpy()                                                 # Calculating the mean of the last hidden state across the first dimension (dim=1)
    job_description_embeddings.append(embeddings)                                                             # Appending the mean embedding to the list


def calculate_cosine_similarity(job_description_embedding, cv_embedding):
  """Calculates the cosine similarity between two vectors."""
  job_description_embedding = job_description_embedding.flatten()                                             # Flattening the embedding for similar array dimenstions of both the embeddings
  cv_embedding = cv_embedding.flatten()
  return np.dot(job_description_embedding, cv_embedding) / (np.linalg.norm(job_description_embedding) * np.linalg.norm(cv_embedding))


job_description_embeddings = np.array(job_description_embeddings)
cv_embeddings = np.array(cv_embeddings)
cosine_similarities = []

for i, job_description_embedding in enumerate(job_description_embeddings):
  for j, extracted_cv_embedding in enumerate(cv_embeddings):
    cosine_similarities.append(calculate_cosine_similarity(job_description_embedding, extracted_cv_embedding))  # Calculation of cosine similarities



cosine_similarities = np.array(cosine_similarities)
cv_ranks = np.argsort(cosine_similarities)
print(cv_ranks)
print(cosine_similarities)
top_5_cvs = []
for i, job_description_ranks in enumerate(cv_ranks):
  if isinstance(cv_details, list):
    # Get the length of the list.
    cv_details_length = len(cv_details)
    top_5_cvs.append([cv_details[cv_rank] for cv_rank in job_description_ranks[:5]])
  else:
     pass

# Print the top 5 CVs for each job description.
for i, job_description in enumerate(df.job_description):
  print(f"Job description: {job_description}")
  print(f"Top 5 CVs:")
  if i < len(top_5_cvs) and len(top_5_cvs[i]) >= 5:
    for cv in top_5_cvs[i]:
      print(f"- {cv}")
  else:
     print("There are no top 5 CVs for this job description.")