itk_module_test()

set(DRRTests
  itkMinimalStandardRandomVariateGeneratorTest.cxx
  itkMyFilterTest.cxx
  )

CreateTestDriver(DRR "${DRR-Test_LIBRARIES}" "${DRRTests}")

itk_add_test(NAME itkMinimalStandardRandomVariateGeneratorTest
  COMMAND DRRTestDriver itkMinimalStandardRandomVariateGeneratorTest
  )

itk_add_test(NAME itkMyFilterTest
  COMMAND DRRTestDriver
    --compare
    DATA{Baseline/itkMyFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkMyFilterTestOutput.mha
  itkMyFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkMyFilterTestOutput.mha
  )
