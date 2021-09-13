from nlp.tools import NlpAnalyzer


class KbqaTemplate(object):
    pass


class KbqaRelationTemplate(KbqaTemplate):
    def __init__(self, _src_entity_id, _relation_id, _obj_entity_id):
        self.src_entity_id = _src_entity_id
        self.relation_id = _relation_id
        self.obj_entity_id = _obj_entity_id


class KbqaPropertyTemplate(KbqaTemplate):
    def __init__(self, _entity_id, _property, _value):
        self.entity_id = _entity_id
        self.property = _property
        self.value = _value


class QueryDao(object):
    def query(self):
        return []


class Engine(object):
    def __init__(self):
        self.__analyzer = NlpAnalyzer()
        self.__query = QueryDao()

    def query(self, text):
        data = self.__analyzer.analyze(text)
        temp_id, params = self.find_intent(data)

        template = self.fill_template(temp_id, None)

        self.exec(template)

    def exec(self, template):
        self.exec_query(template)

    def exec_query(self, template):
        query = self.wrap_query(template)
        results = self.__query.query(query)

        obj_list = []

        if isinstance(template, KbqaRelationTemplate):
            for result in results:
                obj_list.append(KbqaRelationTemplate(result[0], result[1], result[2]))

        elif isinstance(template, KbqaPropertyTemplate):
            for result in results:
                obj_list.append(KbqaPropertyTemplate(result[0], result[1], result[2]))

        return obj_list

    def wrap_query(self, template):
        query = '''SELECT * FROM %s WHERE %s'''

        table = ''
        term = ''
        if isinstance(template, KbqaRelationTemplate):
            table = 'tb_entity_relation'
            if template.src_entity_id is None:
                term += ' obj_entity_id='+template.obj_entity_id
                term += ' AND relation_id='+template.relation_id
            elif template.obj_entity_id is None:
                term += ' src_entity_id=' + template.src_entity_id
                term += ' AND relation_id=' + template.relation_id
            elif template.relation_id is None:
                term += ' obj_entity_id=' + template.obj_entity_id
                term += ' AND src_entity_id=' + template.src_entity_id

        if isinstance(template, KbqaPropertyTemplate):
            table = 'tb_entity_prop'
            if template.value is None:
                term += ' entity_id='+template.entity_id
                term += ' AND property_id='+template.property
            elif template.entity_id is None:
                term += ' property_id=' + template.property
                term += ' AND value=' + template.value

        return query % (table, term)

    def find_entity(self, data):
        word_list = data['word']
        entity_list = data['ent']
        tag_list = data['tag']
        dep_rel_list = data['deprel']

        for i in range(0, len(word_list)):
            ent = entity_list[i]
            if ent is None:
                continue
            word = word_list[i]

    def find_intent(self, data):
        entity_list = self.find_entity(data)

        temp_id = 0
        params = None

        return temp_id, params

    def fill_template(self, temp_id, params):
        return None


if __name__ == '__main__':
    engine = Engine()

    text = '京东方老板江大桥的主营业务都有哪些'
    engine.query(text)

