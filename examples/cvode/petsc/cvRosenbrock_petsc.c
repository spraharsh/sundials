/*
  UNDER WORK
  Probably need to keep it here since
  it depends on implicit SUNDIALS variables until the corresponding interface
  is built.
*/

/*-----------------------------------------------------------------
 *
 *-----------------------------------------------------------------
 * acknowledgement: some code reused from other petsc examples
 *-----------------------------------------------------------------
 * Example/Test to compare function evaluation performance between
 * sundials dense and a comparable petsc implementation using
 * cholesky for a system defined by
 * $$
 * x' = - \grad{V(x)}
 * $$
 *
 * where
 * $$
 * V(x) = - (a-x[0])^2 + b(x[1]-x[0]^2)^2
 * $$
 * This is the Rosenbrock function (not to be confused with
 * rosenbrock ODE methods). The key idea of this test is to
 * identify the attractor (There is only one,the global minimum)
 * corresponding to a point.
 * The key idea is not to solve an optimization problem
 * (for which there are better methods check Nocedal and Wright)
 * but to ensure the PETSc version works just as well to identify
 * the attractor for a slightly non trivial problem where the Jacobian
 * is symmetric.
 *
 * Note: no non convex regions here, maybe a better example might help
 * but all examples I know are slightly non trivial
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2020, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 *---------------------------------------------------------------*/

/* #include <petscerror.h> */
#include <math.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpctypes.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <stdio.h>
#include <string.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sundials/sundials_nvector.h>

/* TODO remove this*/
#include "/home/praharsh/Dropbox/research/bv-libraries/sundials/src/cvode/cvode_impl.h"
static char help[] =
    "CVODE example for finding attractor corresponding to "
    "point using petsc and dense sunmatrix \n";

/*
  Include "cvode.h" for access to the CVODE BDF integrator. Include
  "sunnonlinsol_petscsnes.h" for access to the SUNNonlinearSolver
Unexpected ';' before ')'Unexpected ';' before ')'Unexpected ';' before ')'  wrapper for PETSc SNES.
*/
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_petsc.h>   /* access to the PETSc N_Vector*/
#include <sundials/sundials_math.h>  /* definition of ABS and EXP */
#include <sundials/sundials_types.h> /* definition of type realtype */

#include "sunnonlinsol/sunnonlinsol_petscsnes.h" /* access to the fixed point SUNNonlinearSolver */

/* private constants */
#define ONE RCONST(1.0)
#define TWO RCONST(2.0)

/*****************************************************************************/
/*                 General structs/definitions for later use                 */
/*****************************************************************************/
/* User supplied Jacobian function prototype */
typedef PetscErrorCode (*CVSNESJacFn)(PetscReal t, Vec x, Mat J,
                                      void *user_data);
/*
  context passed on to the delayed hessian
*/
typedef struct {
  /* memory information */
  void *cvode_mem; /* cvode memory */

  /* TODO: remove this since user data is in cvode_mem */
  void *user_mem; /* user data */

  /* jacobian calculation information */
  booleantype jok;   /* check for whether jacobian needs to be updated */
  booleantype *jcur; /* whether to use saved copy of jacobian */
  realtype gamma;
  PetscReal t;

  /* Linear solver, matrix and vector objects/pointers */
  /* NOTE: some of this might be uneccessary since it maybe stored
     in the KSP solver */
  Mat savedJ;                /* savedJ = old Jacobian                        */
  Vec ycur;                  /* CVODE current y vector in Newton Iteration   */
  Vec fcur;                  /* fcur = f(tn, ycur)                           */
  CVSNESJacFn user_jac_func; /* user defined Jacobian function */
  booleantype
      scalesol; /* exposed to user (Check delayed matrix versions later)*/
} * CVLSPETScMem;

/*****************************************************************************/
/*               End general structs/definitions for later use               */
/*****************************************************************************/

/*
  User-defined routines in PETSc TS format
*/
extern PetscErrorCode FormFunction(DM, PetscReal, Vec, Vec, void *);
extern PetscErrorCode FormInitialSolution(DM, Vec);
extern PetscErrorCode MySNESMonitor(SNES, PetscInt, PetscReal,
                                    PetscViewerAndFormat *);

typedef struct {
  PetscReal a; /* a in the standard rosenbrock function definition */
  PetscReal b; /* b in the standard rosenbrock function definition */
} * UserData;

/*
  User-defined routines in CVODE format
*/

/* f - computes f(t,x); this interfaces FormFunction to the CVODE expected
 * format */
extern int f(PetscReal t, N_Vector x, N_Vector xdot, void *ptr);
extern int rosenbrock_gradient(PetscReal t, N_Vector x, N_Vector xdot,
                               void *ptr);
extern PetscErrorCode MyCVodeMonitor(long int, PetscReal, Vec, void *);

/* private helper function for checking return value from SUNDIALS calls */
static int check_retval(void *value, const char *funcname, int opt);

int main(int argc, char **argv) { printf("hello world"); }

/* ------------------------------------------------------------------- */

/* computes f(t,x) in the CVODE expected format. this is (- gradient) of the
   rosenbrock funtion Here t is a dummy variable to parametrize the path */
int rosenbrock_minus_gradient_petsc(PetscReal t, N_Vector x, N_Vector xdot,
                                    void *user_data) {
  /* declarations */
  PetscErrorCode ierr;
  Vec x_petsc = N_VGetVector_Petsc(x);
  Vec xdot_petsc = N_VGetVector_Petsc(xdot);
  UserData data;
  double m_a; /* member a */
  double m_b; /* member b */

  /* get read only array from vector */
  const double *x_arr;
  VecGetArrayRead(x_petsc, &x_arr);

  /* extract information from user data */
  data = (UserData)user_data;
  m_a = data->a;
  m_b = data->b;

  /* initalize to zero*/
  ierr = VecZeroEntries(xdot_petsc);
  CHKERRQ(ierr);
  /* First calculate the gradient \grad{V(x)}*/
  ierr = VecSetValue(xdot_petsc, 0,
                     4 * m_b * x_arr[0] * x_arr[0] * x_arr[0] -
                         4 * m_b * x_arr[0] * x_arr[1] + 2 * m_a * x_arr[0] -
                         2 * m_a,
                     INSERT_VALUES);
  CHKERRQ(ierr);
  ierr = VecSetValue(xdot_petsc, 1, 2 * m_b * (x_arr[1] - x_arr[0] * x_arr[0]),
                     INSERT_VALUES);
  CHKERRQ(ierr);

  /* restore read array */
  ierr = VecRestoreArrayRead(x_petsc, &x_arr);
  CHKERRQ(ierr);

  /* assemble the vector */
  ierr = VecAssemblyBegin(xdot_petsc);
  CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xdot_petsc);
  CHKERRQ(ierr);

  /* scale the vector by -1. since xdot is -\grad{V(x)} */
  ierr = VecScale(xdot_petsc, -1.0);

  CHKERRQ(ierr);
  return (0);
}

/* computes -\grad{f(t,x)} in the CVODE expected format. this is (- hessian) of
   the rosenbrock funtion t in this problem is a dummy variable to parametrize
   path */
PetscErrorCode rosenbrock_minus_Jac_petsc(PetscReal t, Vec x, Mat J,
                                          void *user_data) {
  /* Declarations */
  PetscErrorCode ierr;
  UserData data;
  double m_a;           /* member a */
  double m_b;           /* member b */
  PetscReal hessarr[4]; /* hessian data*/

  /* extract information from user data */
  data = (UserData)user_data;
  m_a = data->a;
  m_b = data->b;

  /* initalize jacobian to zero */
  MatZeroEntries(J);

  /* Illustration of how to deal with symmetric matrices */
  /* get the jacobian type */
  MatType Jac_type;
  MatGetType(J, &Jac_type);

  /* get read only array from vector */
  const double *x_arr;
  VecGetArrayRead(x, &x_arr);

  /* if the hessian is symmetric assume that the matrix is upper triangular */
  if (strcmp(Jac_type, MATSBAIJ) || strcmp(Jac_type, MATSEQSBAIJ) ||
      strcmp(Jac_type, MATMPISBAIJ)) {
    hessarr[0] = 2 + 8 * m_b * x_arr[0] * x_arr[0] -
                 4 * m_b * (-x_arr[0] * x_arr[0] + x_arr[1]);
    hessarr[1] = -4 * m_b * x_arr[0];
    hessarr[2] = 0; /* 0 since symmetric */
    hessarr[3] = 2 * m_b;
  } else {
    hessarr[0] = 2 + 8 * m_b * x_arr[0] * x_arr[0] -
                 4 * m_b * (-x_arr[0] * x_arr[0] + x_arr[1]);
    hessarr[1] = -4 * m_b * x_arr[0];
    hessarr[2] = 0; /* 0 since symmetric */
    hessarr[3] = 2 * m_b;
  }

  /* restore array */
  VecRestoreArrayRead(x, &x_arr);

  PetscInt idxm[] = {0, 1};
  MatSetValues(J, 2, idxm, 2, idxm, hessarr, INSERT_VALUES);
  MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

  /* take the negative since J = -H */
  MatScale(J, -1.0);
  return (0);
}

PetscErrorCode MyCVodeMonitor(long int step, PetscReal ptime, Vec v,
                              void *ctx) {
  PetscErrorCode ierr;
  PetscReal norm;
  PetscFunctionBeginUser;
  ierr = VecNorm(v, NORM_2, &norm);
  CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "timestep %D time %g norm %g\n", step,
                     (double)ptime, (double)norm);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  MySNESMonitor - illustrate how to set user-defined monitoring routine for
  SNES. Input Parameters: snes - the SNES context its - iteration number fnorm
  - 2-norm function value (may be estimated) ctx - optional user-defined
  context for private data for the monitor routine, as set by SNESMonitorSet()
*/
PetscErrorCode MySNESMonitor(SNES snes, PetscInt its, PetscReal fnorm,
                             PetscViewerAndFormat *vf) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SNESMonitorDefaultShort(snes, its, fnorm, vf);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Check function return value...
   opt == 0 means SUNDIALS function allocates memory so check if
   returned NULL pointer
   opt == 1 means SUNDIALS function returns a retval so check if
   retval >= 0
   opt == 2 means function allocates memory so check if returned
   NULL pointer
*/
static int check_retval(void *value, const char *funcname, int opt) {
  int *errretval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && value == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  /* Check if retval < 0 */
  else if (opt == 1) {
    errretval = (int *)value;
    if (*errretval < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *errretval);
      return 1;
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && value == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  return 0;
}

/*****************************************************************************/
/*                      General function for later use                       */
/*****************************************************************************/

/*-----------------------------------------------------------------
  Wrapper to imitate what the linear system solver does in
  cvLsLinSys with what the Jacobian wrapper does in Petsc.

  Calculates a delayed J_{snes} = I - \gamma J_{cvode}

  Warning: In sundials language the Jacobian is the gradient of
  the RHS of the ODE,

  dx/dt = f(x, t)

  to be solved. This is the language we're
  using here. For SNES however the Jacobian is the gradient of
  the LHS of the Nonlinear system

  F(x)=b

  This difference in language is important to keep in mind. since
  this function is called by SNESSetJacobian which refers the
  Jacobian in SNES language.
  Both of these are related by A = J_{SNES} = I-\gamma J_{cvode}
  -----------------------------------------------------------------*/
PetscErrorCode CVDelayedJSNES(SNES snes, Vec X, Mat A, Mat Jpre,
                              void *context) {
  PetscFunctionBegin;
  /* storage for petsc memory */
  CVLSPETScMem cvls_petsc_mem;
  cvls_petsc_mem = (CVLSPETScMem)context;
  /* if solution scaling is not used this is a bad idea */
  if (!cvls_petsc_mem->scalesol) {
    PetscFunctionReturn(0);
  }
  PetscErrorCode ierr;
  /* cvode memory */
  CVodeMem cv_mem;
  int retval;
  realtype gamma;

  CVodeGetCurrentGamma(cvls_petsc_mem->cvode_mem, &gamma);

  cv_mem = (CVodeMem)(cvls_petsc_mem->cvode_mem);

  /* check jacobian needs to be updated */
  if (cvls_petsc_mem->jok) {
    /* use saved copy of jacobian */
    *(cvls_petsc_mem->jcur) = PETSC_FALSE;

    /* Overwrite linear system matrix with saved J
       Assuming different non zero structure */
    /* TODO: expose the NON zero structure usage */
    ierr = MatCopy(cvls_petsc_mem->savedJ, A, DIFFERENT_NONZERO_PATTERN);
    CHKERRQ(ierr);
  } else {
    /* call jac() to update the function */
    *(cvls_petsc_mem->jcur) = PETSC_TRUE;
    ierr = MatZeroEntries(A);
    CHKERRQ(ierr);

    /* compute new jacobian matrix */
    ierr = cvls_petsc_mem->user_jac_func(cvls_petsc_mem->t, X, A,
                                         cvls_petsc_mem->user_mem);
    CHKERRQ(ierr);
    /* Update saved jacobian copy */
    /* TODO: expose nonzero structure usage */
    /* I'm not sure this makes sense */
    MatCopy(A, cvls_petsc_mem->savedJ, DIFFERENT_NONZERO_PATTERN);
  }
  /* do A = I - \gamma J */
  ierr = MatScale(A, -gamma);
  CHKERRQ(ierr);
  ierr = MatShift(A, 1.0);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*-----------------------------------------------------------------
 Pre solve KSP routine. The goal of this function is to emulate what
 is happening before cvLsSolve in the sundials routine
 this should be passed on to KSP from KSPsetpresolve

 should probably call set ksp set reuse preconditioner though
 lets do this change once we get the normal version running

 TODO: THis is not necessary for us since we're planning to go with a
 solver. but iterative solvers need this set up
  -----------------------------------------------------------------*/
PetscErrorCode cvLSPresolveKSP(KSP ksp, Vec b, Vec x, void *ctx) {
  return 0;
}

/*-----------------------------------------------------------------
  Calls the post solve routine for the KSP Solver to emulate the
  functionality of the cvode solver to rescale the step if
  Here the application context is the cvode memory
  -----------------------------------------------------------------*/
PetscErrorCode cvLSPostSolveKSP(KSP ksp, Vec b, Vec x, void *context) {
  /* storage for petsc memory */
  CVLSPETScMem cvls_petsc_mem;
  cvls_petsc_mem = (CVLSPETScMem)context;
  if (!cvls_petsc_mem->scalesol) {
    PetscFunctionReturn(0);
  }

  CVodeMem cv_mem;
  cv_mem = (CVodeMem)context;

  if (cv_mem->cv_gamrat != ONE) {
      VecScale(b,TWO/(ONE + cv_mem->cv_gamrat));
  }
}
