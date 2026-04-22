from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from pickle import load, dump
from sentiment_metrics import compute_all_metrics


def load_dataset():
	with open("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "r", encoding = "latin1") as file:
		data = file.read()
	data = [line.split("@") for line in data.split("\n") if line]
	x = [line[0] for line in data]
	y = [line[1] for line in data]
	# train = 60%, test = 20%, val = 20%
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train, random_state=0)
	return x_train, x_test, x_val, y_train, y_test, y_val


def extract_features(x_train):
	stop_words = [' \'s', 'the', ' (', ' )', ' .',  'herein', 'thereby', 'whereas', 'hereinbefore', 'aforementioned', 'a', 'this', 'that', 'is', 'are', 'be', 'was', 'were', 'and', 'or']	
	vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.5, sublinear_tf=True, stop_words=stop_words)
	x_train = vectorizer.fit_transform(x_train)
	with open("vectorizer.pkl", "wb") as file:
		dump(vectorizer, file)
	return x_train, vectorizer

def train_model(x_train, y_train, x_val, y_val):
	_model = SVC()
	params = {
		"C": [.001, .01, 0.1, 1, 10, 100],
		"kernel": ["linear", "poly", "rbf", "sigmoid"],
		"gamma": [0.1, 1, "scale", "auto"],
		"degree": [2, 3, 4],
		"class_weight": ["balanced"],
		"probability": [True],
		"random_state": [0]
	}
	optimizer = GridSearchCV(_model, params, scoring="balanced_accuracy", n_jobs=-1, verbose=3)
	optimizer.fit(x_val, y_val)
	print(f"Best params: {optimizer.best_params_}")
	model = optimizer.best_estimator_.fit(x_train, y_train)
	with open("model.pkl", "wb") as file:
		dump(model, file)
	return model

def evaluate_model(model, x_test, y_test):
	y_pred = model.predict(x_test)
	print("Balanced accuracy:", round(balanced_accuracy_score(y_test, y_pred), 4))
	print("Precision:", round(precision_score(y_test, y_pred, average="weighted"), 4))
	print("Recall:", round(recall_score(y_test, y_pred, average="weighted"), 4))
	print("F1:", round(f1_score(y_test, y_pred, average="weighted"), 4))
	metrics = compute_all_metrics(
		y_true=y_test,
		y_pred=y_pred,
		inference_records=None
	)
	print("关联指标：", metrics)
if __name__ == "__main__":
	x_train, x_test, x_val, y_train, y_test, y_val = load_dataset()

	try:
		with open("vectorizer.pkl", "rb") as file:
			vectorizer = load(file)

		with open("model.pkl", "rb") as file:
			model = load(file)
   
		x_test = vectorizer.transform(x_test)	
	except FileNotFoundError:
		x_train, vectorizer = extract_features(x_train)
		x_test = vectorizer.transform(x_test)
		x_val = vectorizer.transform(x_val)
		model = train_model(x_train, y_train, x_val, y_val)
	evaluate_model(model, x_test, y_test)
