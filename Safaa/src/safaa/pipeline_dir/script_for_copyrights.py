# SPDX-License-Identifier: CC-BY-SA-4.0
# SPDX-FileCopyrightText: 2025 Abdulsobur Oyewale
# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
from datetime import datetime
import argparse


load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="Path to save the CSV output file")
parser.add_argument("--limit", type=int, default=21000, help="SQL row fetch limit")
args = parser.parse_args()

required_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]
missing = [var for var in required_vars if not os.getenv(var)]

if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


sql_template = """
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
        LIMIT {limit};
        """

def fetch_copyright_data():

    connection = None
    cursor = None
    df = pd.DataFrame()

    try:
        connection = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        cursor =  connection.cursor()

        sql = sql_template.format(limit=args.limit)

        cursor.execute(sql)

        # records = cursor.fetchall()

        batch_size = 20000
        results = []

        while True:
            records = cursor.fetchmany(batch_size)

            if not records:
                break
            for record in records:
                result = {
                    "original_content": record[2],
                    "original_is_enabled": record[7],
                    "edited_content": record[3],
                    "modified_is_enabled": record[8]
                }
                results.append(result)

            df = pd.DataFrame(results)

        timestamp = datetime.now().strftime("%m_%d_%Y")
        default_filename = f"copyrights_{timestamp}.csv"
        output_path = args.output or default_filename
        df.to_csv(output_path, index=False)


    except Exception as error:
        print(f"Error: {error}")

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


if __name__ == "__main__":
    fetch_copyright_data()
