from itertools import islice
from gpt_researcher.retrievers.arxiv.my_arxiv.my_arxiv_retriever import ArxivRetriever


class ArxivSearch:
    """
    ArxivSearch API Retriever
    """
    def __init__(self, query):
        self.arx_ret = ArxivRetriever(load_all_available_meta=True, load_max_docs=3)
        self.query = query

    def search(self, max_results=5):
        """
        Performs the search
        :param query:
        :param max_results:
        :return:
        """
        self.arx_ret.load_max_docs = max_results
        arx_gen = self.arx_ret.get_summaries_as_docs(self.query)
        
        output = []
        for doc in arx_gen:
            output.append({"page_content": doc.page_content,
                           "metadata": doc.metadata})

        return output
    
    def generate_content(self, search_result):
        """function that generates a list of content str.

        Args:
            search_result (_type_): expects a return value of th get_summaries_as_docs func including the following metadata:\n
            "Published": result.updated.date(),
            "Title": result.title,
            "Authors": ", ".join(a.name for a in result.authors),
            "Url": result.entry_id,
            "PDF": result.pdf_url
        """

        content = []
        for document in search_result:
            metadata = document.get("metadata", None)
            if metadata is None:
                continue
            output_dict = {"url" : metadata["Url"],
                            "raw_content":document["page_content"],
                            "title": document["metadata"]["Title"],
                            "published": document["metadata"]["Published"],
                            "authors": document["metadata"]["Authors"]
                            }
                
            content.append(output_dict)
        
        return content

            