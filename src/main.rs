use vector_db::{Vector, VectorDB};

fn main() {
    let mut db = VectorDB::new();

    db.add(vec![1.0, 0.0, 0.0]); // id 0
    db.add(vec![0.0, 1.0, 0.0]); // id 1
    db.add(vec![0.9, 0.1, 0.05]); // id 2 (similar to id 0)
    db.add(vec![-1.0, 0.0, 0.0]); // id 3 (opposite of id 0)
    db.add(vec![0.5, 0.5, 0.5]); // id 4

    println!("DB size: {}", db.len());

    let query: Vector = vec![0.8, 0.2, 0.0];
    let k = 3;
    let results = db.search(&query, k, 1);

    println!("\nTop-{}", k);
    for result in results {
        println!("ID: {}, Score: {}", result.id, result.score);
    }
}
