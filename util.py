from collections import Counter

sign_classes = ['Hello', 'Thank you', 'Ambulance', 'Call', 'Doctor', 'Hurt', 'road']
sign_classes_korean = ['안녕하세요', '감사합니다', '구급차', '불러주세요', '의사', '아파요', 'road']


def get_frequent_output(outputs):
    freq_counter = Counter(outputs)
    common = freq_counter.most_common(2)
    freq_output, cnts = common[0]
    if freq_output == 6:
        if len(common) > 1:
            freq_output2, cnts2 = common[1]
            freq_output = freq_output2
            portion = cnts2 / (len(outputs) - cnts)
            cnts = portion * len(outputs)
        else:
            cnts = -1

    return freq_output, cnts

def get_text(index):
    korean_text = sign_classes_korean[index]
    english_text = sign_classes[index]

    return korean_text, english_text