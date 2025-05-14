from modelGenerate import generate

# {'text': '제1조(목적) 이 조례는 2018 평창 동계올림픽의 성공적 개최 준비를 위하여 "2018성공개최평창군위원회"의 설립 및 활동지원에 관한 사항을 규정함을 목적으로 한다.'
# , 'ner_tag': [{'begin': 19, 'end': 27, 'entity': '평창 동계올림픽', 'id': 1, 'type': 'EV'}
# , {'begin': 29, 'end': 35, 'entity': '성공적 개최', 'id': 2, 'type': 'TM'}
# , {'begin': 0, 'end': 3, 'entity': '제1조', 'id': 3, 'type': 'CV'}
# , {'begin': 14, 'end': 18, 'entity': '2018', 'id': 4, 'type': 'DT'}
# , {'begin': 45, 'end': 49, 'entity': '2018', 'id': 5, 'type': 'DT'}
# , {'begin': 67, 'end': 71, 'entity': '활동지원', 'id': 6, 'type': 'CV'}]}

generate("한국의 인공지능 기술은")