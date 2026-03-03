"""
Configuration for planning cases.
Add new cases here as new plansaker become available.
"""

CASES: dict[str, dict] = {
    "sinsenveien_11": {
        "name": "Sinsenveien 11",
        "description": (
            "Planforslag for Sinsenveien 11 i Oslo. "
            "Saken gjelder regulering og berører naboeiendommer i området."
        ),
        "official_url": "https://innsyn.pbe.oslo.kommune.no/sidinmening/main.asp?idnr=2025078414",
        "case_number": "202506699",
        "municipality": "Oslo",
        "data_dir": "data/sinsenveien_11",
        "vector_store_dir": "vector_store/sinsenveien_11",
        "collection_name": "sinsenveien_11",
    },
}

CONFLICT_TOPICS = [
    "Byggehøyde og utnyttelsesgrad (BYA/TU)",
    "Solforhold og skygge for naboer",
    "Trafikk, parkering og adkomst",
    "Grøntområder, friarealer og lekeplasser",
    "Kulturminner og bevaringshensyn",
    "Støy, luftkvalitet og miljø",
    "Barnehage-, skole- og tjenestekapasitet",
    "Overvann, flom og VA-infrastruktur",
    "Naboers rettigheter og innsigelsesgrunnlag",
]
