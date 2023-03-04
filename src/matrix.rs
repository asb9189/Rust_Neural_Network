use std::ops::Range;
use rand::prelude::*;

pub struct Matrix {
    num_rows: usize,
    num_cols: usize,
    matrix: Vec<Vec<f64>>
}

// Matrix multiplication
// panics when the dimensions are not valid
pub fn mul(m1: &Matrix, m2: &Matrix) {
    if m1.num_cols != m2.num_rows { panic!("Invalid dimensions for matrix multiplication") };

    let mut v: Vec<f64> = vec![];

    // for each row in m1
    for r in 0..(m1.num_rows) {
        for c in 0..(m2.num_cols) {

        }
    }
        // for each column in m2

        // dot product and store result in v

    // after each column in m2 is done, add v into resulting vector
    // and build matrix using from(...)

}

impl Matrix {

    /// Returns zero nr x nc matrix
    pub fn new(nr: usize, nc: usize) -> Matrix {
        let mut matrix: Vec<Vec<f64>> = Vec::new();
        for _ in 0..nr {
            let mut temp: Vec<f64> = Vec::new();
            for _ in 0..nc {
                temp.push(0.0);
            }
            matrix.push(temp);
        };

        Matrix {
            num_rows: nr,
            num_cols: nc,
            matrix: matrix
        }
    }

    // Returns a nr x nc matrix with random values within the given range
    pub fn from_range(nr: usize, nc: usize, range: Range<f64>) -> Matrix {
        let mut matrix: Vec<Vec<f64>> = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..nr {
            let mut temp: Vec<f64> = Vec::new();
            for _ in 0..nc {
                let n = rng.gen_range(range.clone());
                temp.push(n);
            }
            matrix.push(temp);
        };

        Matrix {
            num_rows: nr,
            num_cols: nc,
            matrix: matrix
        }
    }

    /// Panics if dimensions are not valid
    pub fn from(m: Vec<Vec<f64>>) -> Matrix {

        let nr = m.len();
        if nr == 0 { panic!("Vector has no rows") }

        // Get length of first row
        let nc = m[0].len();
        if nc == 0 { panic!("Vector has no columns") }

        // Check that each row has the same length
        for r in 0..nr { if m[r].len() != nc{ panic!("Vector dimensions are invalid") } };

        Matrix {
            num_rows: nr,
            num_cols: nc,
            matrix: m,
        }
    }

    pub fn get_row(&self, row_index: usize) -> Vec<f64> {
        let mut v: Vec<f64> = vec![];

        if row_index >= self.num_rows { panic!("row index out of bounds") };
        for c in 0..(self.num_cols) {
            v.push(self.matrix[row_index][c]);
        };
            
        v
    }

    pub fn get_column(&self, column_index: usize) -> Vec<f64> {
        let mut v: Vec<f64> = vec![];

        if column_index >= self.num_cols { panic!("column index out of bounds") };
        for r in 0..(self.num_rows) {
            v.push(self.matrix[r][column_index]);
        };
            
        v
    }

    pub fn display(&self) {
        let nr = self.num_rows;
        let nc = self.num_cols;

        print!("{nr}x{nc} Matrix\n");
        for r in 0..nr {
            for c in 0..nc {
                let v = self.matrix[r][c];
                print!("{v} ");
            }
            println!();
        }
        println!();
    }
}
