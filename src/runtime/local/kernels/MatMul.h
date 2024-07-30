/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/CastObj.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct MatMul {
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool transa, bool transb, DCTX(ctx)) = delete;
};

template<typename T>
struct MatMul<DenseMatrix<T>, DenseMatrix<T>, DenseMatrix<T>> {
    static void apply(DenseMatrix<T> *&res, const DenseMatrix<T> *lhs, const DenseMatrix<T> *rhs, bool transa,
                      bool transb, DCTX(dctx));
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void matMul(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool transa, bool transb, DCTX(ctx)) {
    MatMul<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, transa, transb, ctx);
}

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct MatMul<DenseMatrix<VT>, CSRMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const CSRMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, bool transa, bool transb, DCTX(ctx)) {
        const size_t nr1 = lhs->getNumRows();
        [[maybe_unused]] const size_t nc1 = lhs->getNumCols();

        [[maybe_unused]] const size_t nr2 = rhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();

        if (nc1 != nr2) {
            throw std::runtime_error(
                "MatMul - #cols of lhs and #rows of rhs must be the same");
        }
        // FIXME: transpose isn't supported atm

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, false);

        const VT * valuesRhs = rhs->getValues();
        VT * valuesRes = res->getValues();

        const size_t rowSkipRhs = rhs->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        memset(valuesRes, VT(0), sizeof(VT) * nr1 * nc2);
        for(size_t r = 0; r < nr1; r++) {
            const size_t rowNumNonZeros = lhs->getNumNonZeros(r);
            const size_t * rowColIdxs = lhs->getColIdxs(r);
            const VT * rowValues = lhs->getValues(r);

            const size_t rowIdxRes = r * rowSkipRes;
            for(size_t i = 0; i < rowNumNonZeros; i++) {
                const size_t c = rowColIdxs[i];
                const size_t rowIdxRhs = c * rowSkipRhs;

                for(size_t j = 0; j < nc2; j++) {
		            valuesRes[rowIdxRes + j] += rowValues[i] * valuesRhs[rowIdxRhs + j];
                }
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct MatMul<Matrix<VT>, Matrix<VT>, Matrix<VT>> {
    static void apply(Matrix<VT> *& res, const Matrix<VT> * lhs, const Matrix<VT> * rhs, bool transa, bool transb, DCTX(ctx)) {
        const size_t lhsRows = transa ? lhs->getNumCols() : lhs->getNumRows();
        const size_t lhsCols = transa ? lhs->getNumRows() : lhs->getNumCols();
        const size_t rhsRows = transb ? rhs->getNumCols() : rhs->getNumRows();
        const size_t rhsCols = transb ? rhs->getNumRows() : rhs->getNumCols();
        
        if (lhsCols != rhsRows)
            throw std::runtime_error("MatMul: #cols of lhs and #rows of rhs must be the same");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(lhsRows, rhsCols, false);

        res->prepareAppend();
        for (size_t rowRes = 0; rowRes < lhsRows; ++rowRes) {
            for (size_t colRes = 0; colRes < rhsCols; ++colRes) {
                VT resVal = 0;
                for (size_t cell = 0; cell < lhsCols; ++cell) {
                    VT lhsVal = transa ? lhs->get(cell, rowRes) : lhs->get(rowRes, cell);
                    VT rhsVal = transb ? rhs->get(colRes, cell) : rhs->get(cell, colRes);
                    resVal += lhsVal * rhsVal;
                }
                res->append(rowRes, colRes, resVal);
            }
        }
        res->finishAppend();
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix, CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct MatMul<CSRMatrix<VT>, CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const CSRMatrix<VT> * lhs, const CSRMatrix<VT> * rhs, bool transa, bool transb, DCTX(ctx)) {
      // 1. check the size of the matrices
      const size_t nr1 = lhs->getNumRows();
      [[maybe_unused]] const size_t nc1 = lhs->getNumCols();

      [[maybe_unused]] const size_t nr2 = rhs->getNumRows();
      const size_t nc2 = rhs->getNumCols();

      size_t estimationNumNonZeros = lhs->getNumNonZeros() * rhs->getNumNonZeros();
      if(res == nullptr)
        res = DataObjectFactory::create<CSRMatrix<VT>>(nr1, nc2, estimationNumNonZeros, true);

      assert(nc1 == nr2 && "#cols of lhs and #rows of rhs must be the same");

      // 2. Transpose rhs
      const size_t numRows = rhs->getNumRows();
      const size_t numCols = rhs->getNumCols();

      CSRMatrix<VT>* rhsT = DataObjectFactory::create<CSRMatrix<VT>>(numCols, numRows, rhs->getNumNonZeros(), false);

      const VT * valuesArg = rhs->getValues();
      const size_t * colIdxsArg = rhs->getColIdxs();
      const size_t * rowOffsetsArg = rhs->getRowOffsets();

      const size_t numNonZeros = rhs->getNumNonZeros();

      VT * valuesRhsT = rhsT->getValues();
      size_t * colIdxsRhsT = rhsT->getColIdxs();
      size_t * rowOffsetsRhsT = rhsT->getRowOffsets();

      // Maybe use memset instead?
      // memset(rowOffsetsRhsT, 0, sizeof(size_t) * (numCols + 1));
      for (size_t i = 0; i < numCols + 1; i++)
        rowOffsetsRhsT[i] = 0;

      for (size_t row = 0; row < numRows; row++)
        for (size_t j = rowOffsetsArg[row]; j < rowOffsetsArg[row + 1]; j++)
          rowOffsetsRhsT[colIdxsArg[j]]++;

      for (size_t col = 0, cumsum = 0; col < numCols; col++) {
        size_t tmp = rowOffsetsRhsT[col];
        rowOffsetsRhsT[col] = cumsum;
        cumsum += tmp;
      }
      rowOffsetsRhsT[numCols] = numNonZeros;

      for (size_t row = 0; row < numRows; row++) {
        for (size_t j = rowOffsetsArg[row]; j < rowOffsetsArg[row + 1]; j++) {
          size_t col = colIdxsArg[j];
          size_t dest = rowOffsetsRhsT[col];
          colIdxsRhsT[dest] = row;
          valuesRhsT[dest] = valuesArg[j];
          rowOffsetsRhsT[col]++;
        }
      }

      for (size_t col = 0, last = 0; col < numCols + 1; col++) {
        size_t tmp = rowOffsetsRhsT[col];
        rowOffsetsRhsT[col] = last;
        last = tmp;
      }
      // const size_t numRows = rhs->getNumRows();
      // const size_t numCols = rhs->getNumCols();

      // CSRMatrix<VT>* rhsT = DataObjectFactory::create<CSRMatrix<VT>>(numCols, numRows, rhs->getNumNonZeros(), false);

      // const VT * valuesArg = rhs->getValues();
      // const size_t * colIdxsArg = rhs->getColIdxs();
      // const size_t * rowOffsetsArg = rhs->getRowOffsets();

      // VT * valuesRes = rhsT->getValues();
      // VT * const valuesResInit = valuesRes;
      // size_t * colIdxsRes = rhsT->getColIdxs();
      // size_t * rowOffsetsRes = rhsT->getRowOffsets();

      // auto* curRowOffsets = new size_t[numRows + 1];
      // memcpy(curRowOffsets, rowOffsetsArg, (numRows + 1) * sizeof(size_t));

      // rowOffsetsRes[0] = 0;
      // for(size_t c = 0; c < numCols; c++) {
      //   for(size_t r = 0; r < numRows; r++)
      //     if(curRowOffsets[r] < rowOffsetsArg[r + 1] && colIdxsArg[curRowOffsets[r]] == c) {
      //       *valuesRes++ = valuesArg[curRowOffsets[r]];
      //       *colIdxsRes++ = r;
      //       curRowOffsets[r]++;
      //     }
      //   rowOffsetsRes[c + 1] = valuesRes - valuesResInit;
      // }

      // delete[] curRowOffsets;

      // 3. Compute Matrix Multiplication
      const VT * valuesLhs = lhs->getValues();
      const size_t * colIdxsLhs = lhs->getColIdxs();
      const size_t * rowOffsetsLhs = lhs->getRowOffsets();

      const VT * valuesRhs = rhsT->getValues();
      const size_t * colIdxsRhs = rhsT->getColIdxs();
      const size_t * rowOffsetsRhs = rhsT->getRowOffsets();

      for (size_t row = 0; row < nr1; row++) {
        for (size_t col = 0; col < nc2; col++) {
          size_t i = rowOffsetsLhs[row];
          size_t j = rowOffsetsRhs[col];
          VT sum = VT(0);
          while (i < rowOffsetsLhs[row + 1] && j < rowOffsetsRhs[col + 1]) {
            if (colIdxsLhs[i] == colIdxsRhs[j]) {
              sum += valuesLhs[i] * valuesRhs[j];
              i++; j++;
            } else if (colIdxsLhs[i] < colIdxsRhs[j]) {
              i++;
            } else {
              j++;
            }
          }
          res->set(row, col, sum);
        }
      }
  }
};
