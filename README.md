# NLP-all-in-one

#bert model for sentiment analysis
#Sentiment_analysis_model= "https://drive.google.com/uc?id=1--eULExMNhEKGiY4zZmdSB7dvMwh0nOX"
gcloud builds submit --tag gcr.io/nlp-all-in-ons/nlpallio  --project=nlp-all-in-ons

gcloud run deploy --image gcr.io/nlp-all-in-ons/nlpallio --platform managed  --project=nlp-all-in-ons --allow-unauthenticated
