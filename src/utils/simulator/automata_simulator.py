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


class AutomataSimilator:
    store_id = '吴江新华书店'
    def __init__(self, template):
        self._load_template(template)
        self.IP = "localhost"
        self.core = 'automata'
        self.solr_client = SolrClient('http://{}:11403/solr'.format(self.IP))

    def _load_template(self, template_file):
        self.templates = {}
        self.contents = []
        with open(template_file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                if line and not line.startswith('//'):
                    tokens = line.split('#')
                    if tokens[1] == 'template':
                        unfold = self.unfold_template(tokens[2])
                        template_id = tokens[0]
                        if template_id in self.templates:
                            self.templates[template_id].extend(unfold)
                        else:
                            self.templates[template_id] = unfold
                    if tokens[1] == 'content':
                        content = {}
                        content_id = tokens[0]
                        trigger = tokens[2]
                        entities = tokens[3]
                        template_id = tokens[4]
                        synonyms = tokens[5].split('/')
                        content['intent'] = trigger
                        content['content_id'] = content_id
                        content['entities'] = entities
                        content['template_id'] = template_id.split(',')
                        content['synonyms'] = synonyms
                        self.contents.append(content)

    def simulate(self, content_uids=[], templates=[]):
        for content in self.contents:
            selected_templates = np.concatenate([self.templates[_id] for _id in content['template_id']], axis=0)
            for entity in content['synonyms']:
                for t in selected_templates:
                    line = t.replace('${content}', entity)
                    intent = content['intent']
                    entities = content['entities']
                    uid = intent + '#' + entities
                    if len(content_uids) > 0:
                        if uid not in content_uids:
                            continue
                    if len(templates) > 0:
                        if t not in templates:
                            continue
                    yield (line, intent, entities)

    def update_solr(self, content_uids=[], templates=[]):
        for line, intent, entities in self.simulate(content_uids=content_uids, templates=templates):
            uid = intent + '|' + entities
            r = self.solr_client.delete_doc_by_query(self.core, "question_str:" + line)

            doc = {
                "uid": intent + '|' + entities,
                "intent": intent,
                "question": [line],
                "entities": entities,
                "store_id": self.store_id,
                "media":'null',
                'emotion': 'null',
                "category":"process"
            }
            line = json.dumps(doc, ensure_ascii=False)
            line = "[" + line + "]"
            print(line)
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
    a = AutomataSimilator('automata_template.txt')
    a.update_solr()
    # print(a.unfold_template('你好[我来|我要|我想|有没有|怎么|]<content>[吧|呢|啊|有没有|]'))
