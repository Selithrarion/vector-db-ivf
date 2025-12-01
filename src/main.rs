use anyhow::{Context, Result};
use seli_vector_db::{Vector, VectorDB};

fn main() -> Result<()> {
    let db_path = "vector_db.bin";
    let db = match VectorDB::load_from_file(db_path) {
        Ok(db) => {
            println!("Loaded DB from '{}', size: {}", db_path, db.len());
            db
        }
        Err(_) => {
            println!("No existing DB found. Creating a new one.");
            let mut db = VectorDB::new();
            db.add(vec![1.0, 0.0, 0.0]); // id 0
            db.add(vec![0.0, 1.0, 0.0]); // id 1
            db.add(vec![0.9, 0.1, 0.05]); // id 2 (similar to id 0)
            db.add(vec![-1.0, 0.0, 0.0]); // id 3 (opposite of id 0)
            db.add(vec![0.5, 0.5, 0.5]); // id 4
            db
        }
    };

    let query: Vector = vec![0.8, 0.2, 0.1];
    let k = 3;
    let results = db.search(&query, k, 1);

    println!("\nSearch results for top-{}", k);
    for result in results {
        println!("ID: {}, Score: {}", result.id, result.score);
    }

    db.save_to_file(db_path)
        .context(format!("Failed to save DB to '{}'", db_path))?;
    println!("Successfully saved DB to '{}'", db_path);

    Ok(())
}
