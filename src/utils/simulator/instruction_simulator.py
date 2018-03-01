"""
-------------------------------------------------
   File Name：     instruction_simulator
   Description :
   Author :       deep
   date：          18-1-23
-------------------------------------------------
   Change Activity:
                   18-1-23:
                   
   __author__ = 'deep'
-------------------------------------------------
"""

import sys
import os
import json

import numpy as np
import urllib

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, parentdir)

from SolrClient import SolrClient


class InstructionSimilator:
    store_id = '吴江新华书店'
    def __init__(self, template):
        self._load_template(template)
        self.IP = "localhost"
        self.core = 'instruction'
        self.solr_client = SolrClient('http://{}:11403/solr'.format(self.IP))

    def _load_template(self, template_file):
        self.docs = []
        self.template = {}
        with open(template_file, 'r') as f:
            for line in f:
                if line.startswith('<t>'):
                    tokens = line.strip('\n').split('#')
                    id_ = tokens[1]
                    template = tokens[2]
                    if id_ not in self.template:
                        self.template[id_] = [template]
                    self.template[id_].append(template)
                else:
                    try:
                        tokens = line.strip('\n').split('#')
                        instruction = tokens[0]
                        entities = tokens[1]
                        media = tokens[2]
                        emotion = tokens[3]
                        answer = tokens[4]
                        doc = dict()
                        doc["instruction"] = instruction
                        doc['entities'] = entities
                        doc['media'] = media
                        doc['emotion'] = emotion
                        if answer.startswith('api_call_refer'):
                            answers = self.refer_answer(answer)
                        else:
                            answers = [answer]
                        doc['answer'] = answers
                        doc['store_id'] = self.store_id
                        doc['uid'] = instruction + '|' + entities
                        doc['category'] = 'process'
                        self.docs.append(doc)
                    except:
                        print(line)
                        exit(-1)

    def refer_answer(self, line):
        ids = line.replace('api_call_refer_', '').split(',')
        answers = []
        for id_ in ids:
            templates = self.template[id_]
            answers.extend(templates)
        return answers

    def update_solr(self):
        for doc in self.docs:
            uid = doc["instruction"] + '|' + doc["entities"]
            r = self.solr_client.delete_doc_by_query(self.core, "uid:" + '"' + uid + '"')
            line = json.dumps(doc, ensure_ascii=False)
            line = "[" + line + "]"

            line = str.encode(line)
            req = urllib.request.Request(url='http://{}:11403/solr/{}/update?commit=true'.format(self.IP, self.core),
                                         data=line)
            headers = {"content-type": "text/json"}
            req.add_header('Content-type', 'application/json')
            f = urllib.request.urlopen(req)
            # Begin using data like the following
            f.read()

    def unfold_template(self, line):
        expressions_container = []
        expressions = []
        expression = ''
        flag = False
        for c in line:
            if c == '[':
                flag = True
                if len(expression) > 0:
                    expressions.append(expression)
                if len(expressions) > 0:
                    expressions_container.append(expressions)
                expressions = list()
                expression = ''
                continue
            if c == ']':
                flag = False
                expressions.append(expression)
                if len(expressions) > 0:
                    expressions_container.append(expressions)
                expressions = list()
                expression = ''
                continue
            if flag and c == '|':
                expressions.append(expression)
                expression = ''
                continue
            expression += c

        if len(expression) > 0:
            expressions.append(expression)
            expressions_container.append(expressions)

        return self._unfold(expressions_container)

    def _unfold(self, expressions_container):
        if len(expressions_container) == 0:
            return None
        templates = self._unfold(expressions_container[1:])
        init = expressions_container[0]
        if templates:
            init = expressions_container[0]
            new_templates = []
            for t in init:
                for tt in templates:
                    new_templates.append(t + tt)
            return new_templates
        return init


if __name__ == '__main__':
    a = InstructionSimilator('instructions.txt')
    a.update_solr()
    # print(a.unfold_template('你好[我来|我要|我想|有没有|怎么|]<content>[吧|呢|啊|有没有|]'))

