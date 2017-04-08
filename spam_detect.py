from naive_bayes import NaiveBayes
import os
import email.parser
import lxml.html
import csv


def extract_body(spam_email_text):
    msg = email.message_from_file(spam_email_text)
    payload = msg.get_payload()
    if type(payload) == type(list()):
        payload = payload[0]
    try:
        plain_text_body_content = lxml.html.document_fromstring(str(payload)).text_content()
    except lxml.etree.XMLSyntaxError as e:
        print(e)
        return ""
    return plain_text_body_content


class SpamDetector:
    def __init__(self, spam_emails_path, ham_emails_path, unknown_emails_path):
        # 0 - No spam (Ham)
        # 1 - Spam
        self.naive_bayes = NaiveBayes([0, 1])
        self.spam_emails_path = spam_emails_path
        self.ham_emails_path = ham_emails_path
        self.unknown_emails_path = unknown_emails_path

    def train(self):
        for spam_email in os.listdir(self.spam_emails_path):
            with open(os.path.join(self.spam_emails_path, spam_email), 'r', encoding='utf-8', errors='ignore') \
                    as spam_email_text:
                text = extract_body(spam_email_text)
                self.naive_bayes.train(1, text)

        for ham_email in os.listdir(self.ham_emails_path):
            with open(os.path.join(self.ham_emails_path, ham_email), 'r', encoding='utf-8', errors='ignore') \
                    as ham_email_text:
                text = extract_body(ham_email_text)
                self.naive_bayes.train(0, text)

    def classify(self):
        results = []
        for unknown_email in os.listdir(self.unknown_emails_path):
            with open(os.path.join(self.unknown_emails_path, unknown_email), 'r', encoding='utf-8', errors='ignore') \
                    as unknown_email_text:
                text = extract_body(unknown_email_text)
                res = self.naive_bayes.classify(text)
                results.append({'filename': unknown_email, 'prediction': res})
                # print(res)
        top_features = self.naive_bayes.get_stat()[:31]
        self._write_results(results, top_features)

    def _write_results(self, results, top_features):
        with open('results.csv', 'w+') as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=['id', 'prediction'])
            writer.writeheader()
            for result in results:
                writer.writerow({'id': result['filename'], 'prediction': result['prediction']})

        with open('top_features.csv', 'w+') as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=['feature', 'weight'])
            writer.writeheader()
            for feature in top_features:
                writer.writerow({'feature': feature[0], 'weight': float(feature[1])})

    def cross_check(self):
        spam_res = []
        for spam_email in os.listdir(self.spam_emails_path):
            with open(os.path.join(self.spam_emails_path, spam_email), 'r', encoding='utf-8', errors='ignore') \
                    as spam_email_text:
                text = extract_body(spam_email_text)
                res = self.naive_bayes.classify(text)
                if res != 1:
                    spam_res.append([spam_email, self.naive_bayes.classify(text)])
        print("Spam wasn't detected in {}".format(len(spam_res)))
        print(spam_res)

        ham_res = []
        for ham_email in os.listdir(self.ham_emails_path):
            with open(os.path.join(self.ham_emails_path, ham_email), 'r', encoding='utf-8', errors='ignore') \
                    as ham_email_text:
                text = extract_body(ham_email_text)
                res = self.naive_bayes.classify(text)
                if res != 0:
                    ham_res.append([ham_email, self.naive_bayes.classify(text)])
        print("Not spam wasn't detected in {}".format(len(ham_res)))
        print(ham_res)


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    detector = SpamDetector(os.path.join(path, 'data', 'spam'),
                            os.path.join(path, 'data', 'notSpam'),
                            os.path.join(path, 'data', 'unknown'))
    detector.train()
    detector.classify()
    detector.cross_check()
