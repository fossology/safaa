# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
from datetime import datetime

load_dotenv()

sql = """
        SELECT DISTINCT ON (C.copyright_pk, UT.uploadtree_pk)
            C.copyright_pk,
            UT.uploadtree_pk AS uploadtree_pk,
            C.content AS original_content,
            CE.content AS edited_content,
            C.hash AS original_hash,
            CE.hash AS edited_hash,
            C.agent_fk AS agent_fk,
            C.is_enabled AS original_is_enabled,
            CE.is_enabled AS modified_is_enabled
        FROM copyright C
        INNER JOIN uploadtree UT ON C.pfile_fk = UT.pfile_fk
        LEFT JOIN copyright_event CE ON CE.copyright_fk = C.copyright_pk
            AND CE.uploadtree_fk = UT.uploadtree_pk
        WHERE C.content IS NOT NULL
            AND C.content != ''
        ORDER BY C.copyright_pk, UT.uploadtree_pk, C.content, CE.content DESC
        LIMIT 21000;
        """

def fetch_copyright_data():
    try:
        connection = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        cursor =  connection.cursor()

        cursor.execute(sql)


        records = cursor.fetchall()
        timestamp = datetime.now().isoformat()
        # print(records)
        results = []
        for record in records:
            result = {
                "original_content": record[2],
                "original_is_enabled": record[7],
                "edited_content": record[3],
                "modified_is_enabled": record[8],
                "fetched_at": timestamp
            }
            results.append(result)
        df = pd.DataFrame(results)



    except Exception as error:
        print(f"Error: {error}")
    finally:
        # print(df)
        # Save the dataframe to a CSV file
        df.to_csv('copyrights.csv', index=False)
        if connection:
            cursor.close()
            connection.close()

fetch_copyright_data()